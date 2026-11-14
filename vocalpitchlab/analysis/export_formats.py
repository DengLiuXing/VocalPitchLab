"""Export analysed notes and pitch into a Standard MIDI File or an OpenUtau project."""
import json
import struct

import numpy as np

PPQ=480
# Cents are measured from A4, the same origin OpenUtau's own import uses.
A4=69
# UCurve.interval: the smallest step a curve point may take, in ticks.
CURVE_INTERVAL=5
PITD_RANGE=1200
# Trim each note edge by this much so OpenUtau can draw its own note transition.
EDGE_TRIM_MS=25.
EDGE_TRIM_RATIO=.15
GAP_FILL_STEPS=12
MEDIAN_RADIUS=2
SPIKE_CENTS=75.
SPIKE_BLEND=.7
# Frame-to-frame jitter is not vibrato; smooth it before it reaches the curve.
SMOOTH_SIGMA_MS=20.
# A new OpenUtau note carries two anchors at +-40 ms.
ANCHOR_MS=40.
# Pitch bend is normalised over +-2 semitones, and the RPN must say so in semitones.
BEND_SEMITONES=2
VELOCITY=100
PROGRAM=54
LYRIC='a'


def _grid(curve):
    """Pitch samples as (per-frame cents or None, hop seconds, first timestamp).

    The analysis stores midi_float; both writers measure a note tone in cents
    from A4, so the conversion happens once here.
    """
    times,values=curve
    times=np.asarray(times,float).ravel();values=np.asarray(values,float).ravel()
    if times.size!=values.size:raise ValueError('Curve length mismatch')
    if times.size<2:return [],.01,0.
    hop=float(np.median(np.diff(times)))
    if not np.isfinite(hop) or hop<=0:raise ValueError('Invalid curve grid')
    return [100.*(float(value)-A4) if np.isfinite(value) else None for value in values],hop,float(times[0])


def _frame(seconds,hop,origin):
    return int(round((seconds-origin)/hop))


def _moment(seconds,bpm):
    return int(round(seconds*PPQ*bpm/60.))


def _span(notes):
    """Notes sorted by start with integer pitch, as the exporters need them."""
    ordered=[]
    for note in sorted(notes,key=lambda item:float(item['start'])):
        start=float(note['start']);end=float(note['end'])
        if end<=start:continue
        ordered.append((int(round(float(note['midi']))),start,end))
    return ordered


def _vlq(value):
    out=bytearray([value&0x7f]);value>>=7
    while value:
        out.append(0x80|(value&0x7f));value>>=7
    return bytes(reversed(out))


def _meta(kind,payload):
    return bytes([0xff,kind])+_vlq(len(payload))+payload


def to_midi(notes,curve,*,bpm=120.):
    """Standard MIDI File bytes.

    One long note per event plus a pitch bend per curve frame, so vibrato and
    glides survive instead of being quantised into short notes. The bend range
    is written as an RPN because the MIDI default would reinterpret our values.
    """
    if bpm<=0:raise ValueError('bpm must be positive')
    samples,hop,origin=_grid(curve)
    events=[]
    # Appending in time order plus a stable sort keeps note_on ahead of the
    # bends of the same tick, and note_off after them.
    events.append((0,_meta(0x51,int(round(60000000./bpm)).to_bytes(3,'big'))))
    events.append((0,bytes([0xc0,PROGRAM])))
    for control,value in ((101,0),(100,0),(6,BEND_SEMITONES),(38,0)):
        events.append((0,bytes([0xb0,control,value])))
    for number,start,end in _span(notes):
        if not 0<=number<=127:continue
        begin=max(0,_moment(start,bpm));finish=_moment(end,bpm)
        if finish<=begin:continue
        events.append((begin,bytes([0x90,number,VELOCITY])))
        if samples:
            base=100.*(number-69)
            first=max(0,_frame(start,hop,origin));last=_frame(end,hop,origin)
            for index in range(first,min(last,len(samples)-1)+1):
                value=samples[index]
                if value is None:continue
                bend=int(round((value-base)/(100.*BEND_SEMITONES)*8192))
                bend=max(-8192,min(8191,bend))+8192
                tick=_moment(origin+index*hop,bpm)
                events.append((tick,bytes([0xe0,bend&0x7f,(bend>>7)&0x7f])))
        events.append((finish,bytes([0x80,number,0])))
    events.sort(key=lambda item:item[0])
    data=bytearray();previous=0
    for tick,message in events:
        # A frame grid rounding down one tick must not push later events later.
        data+=_vlq(max(0,tick-previous))+message
        previous=max(previous,tick)
    data+=_vlq(0)+_meta(0x2f,b'')
    return b'MThd'+struct.pack('>IHHH',6,0,1,PPQ)+b'MTrk'+struct.pack('>I',len(data))+bytes(data)


def _quote(text):
    return json.dumps(text,ensure_ascii=False)


def _interpolate(samples):
    """Fill unvoiced frames linearly, matching OpenUtau's own RMVPE import.

    Dropping the gaps splits the curve into fragments that OpenUtau then joins
    with straight lines, which turns onsets and breaths into sudden glides.
    """
    result=list(samples)
    total=len(result)
    first=next((index for index,value in enumerate(result) if value is not None),None)
    if first is None:return result
    for index in range(first):result[index]=result[first]
    previous=first;index=first+1
    while index<total:
        if result[index] is not None:
            previous=index;index+=1;continue
        gap=index
        while index<total and result[index] is None:index+=1
        if index<total:
            left,right=result[previous],result[index];span=index-previous
            for offset in range(1,span):result[previous+offset]=left+(right-left)*offset/span
            previous=index
        else:
            for tail in range(gap,total):result[tail]=result[previous]
    return result


def _smooth(samples,hop):
    """Gaussian low pass over the voiced frames, normalised by the taps used."""
    sigma=SMOOTH_SIGMA_MS/1000./hop
    if sigma<=0 or len(samples)<3:return list(samples)
    radius=int(np.ceil(3.*sigma))
    kernel=[float(np.exp(-.5*(offset/sigma)**2)) for offset in range(-radius,radius+1)]
    smoothed=[];total=len(samples)
    for index,value in enumerate(samples):
        if value is None:
            smoothed.append(None);continue
        acc=0.;weight=0.
        for offset,tap in zip(range(-radius,radius+1),kernel):
            neighbour=index+offset
            if 0<=neighbour<total and samples[neighbour] is not None:
                acc+=tap*samples[neighbour];weight+=tap
        smoothed.append(acc/weight)
    return smoothed


def _median(values):
    result=[]
    for index in range(len(values)):
        window=sorted(values[max(0,index-MEDIAN_RADIUS):index+MEDIAN_RADIUS+1])
        result.append(window[len(window)//2])
    return result


def _spikes(values):
    if len(values)<=2:return list(values)
    result=list(values)
    for index in range(1,len(values)-1):
        delta=result[index]-(result[index-1]+result[index+1])/2.
        if abs(delta)>SPIKE_CENTS:result[index]=int(round(result[index]-delta*SPIKE_BLEND))
    return result


def _fill_gaps(points):
    if not points:return points
    expanded=[points[0]]
    for point in points[1:]:
        previous=expanded[-1]
        steps=max(0,(point[0]-previous[0])//CURVE_INTERVAL-1)
        if 0<steps<=GAP_FILL_STEPS:
            for step in range(1,steps+1):
                ratio=step/(steps+1)
                expanded.append((previous[0]+step*CURVE_INTERVAL,int(round(previous[1]+(point[1]-previous[1])*ratio))))
        expanded.append(point)
    return expanded


def _simplify(points):
    if len(points)<3:return points
    result=[points[0]]
    for index in range(1,len(points)-1):
        x0,y0=result[-1];x1,y1=points[index];x2,y2=points[index+1]
        if (y1-y0)*(x2-x1)!=(y2-y1)*(x1-x0):result.append((x1,y1))
    result.append(points[-1])
    return result


def _pitd(samples,notes,hop,origin,bpm):
    """Pitch offsets in cents against each note's own tone, as (xs, ys) in ticks."""
    samples=_smooth(_interpolate(samples),hop)
    groups=[]
    for number,start,end in notes:
        length=end-start;edge=min(EDGE_TRIM_MS,length*1000.*EDGE_TRIM_RATIO)
        first=max(0,_frame(start,hop,origin));last=_frame(end,hop,origin)
        group=[]
        for index in range(first,min(last,len(samples)-1)+1):
            value=samples[index]
            if value is None:continue
            offset=(origin+index*hop-start)*1000.
            if length*1000.>edge*2 and (offset<edge or length*1000.-offset<=edge):continue
            x=int(round(_moment(origin+index*hop,bpm)/CURVE_INTERVAL))*CURVE_INTERVAL
            y=int(round(value-100.*(number-69)))
            y=max(-PITD_RANGE,min(PITD_RANGE,y))
            if group and group[-1][0]==x:group[-1]=(x,y)
            elif not group or x>group[-1][0]:group.append((x,y))
        if len(group)>=2:groups.append(group)
    points=[]
    for group in groups:points.extend(_fill_gaps(group))
    if not points:return [],[]
    values=_spikes(_median([y for _,y in points]))
    points=_simplify([(x,y) for (x,_),y in zip(points,values)])
    return [x for x,_ in points],[y for _,y in points]


def _anchors():
    # The note's own pitch stays flat: the real contour lives in pitd, and both
    # would otherwise stack. Anchors plus snap_first still give OpenUtau the
    # short transition line it draws by default.
    return [(-ANCHOR_MS,0.),(ANCHOR_MS,0.)]


def to_ustx(notes,curve,*,bpm=120.,name='Vocal'):
    """OpenUtau project text.

    Every key here is snake_case and verified against OpenUtau's own loader:
    parts live under voice_parts, track_no must resolve, each note needs a pitch
    with at least one point and a non-null lyric, and tempos[0].position plus
    time_signatures[0].bar_position must both be 0.
    """
    if bpm<=0:raise ValueError('bpm must be positive')
    samples,hop,origin=_grid(curve)
    ordered=_span(notes)
    blocks=[];last_end=0
    for index,(number,start,end) in enumerate(ordered):
        position=_moment(start,bpm)
        limit=_moment(ordered[index+1][1],bpm) if index+1<len(ordered) else None
        finish=_moment(end,bpm)
        if limit is not None:finish=min(finish,limit)
        duration=max(10,finish-position)
        blocks+=['  - position: %d'%position,'    duration: %d'%duration,'    tone: %d'%number,
                 '    lyric: %s'%_quote(LYRIC),'    tuning: 0','    pitch:','      data:']
        for x,y in _anchors():blocks.append('      - {x: %g, y: %g, shape: io}'%(x,y))
        blocks.append('    snap_first: true')
        blocks.append('    vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}')
        last_end=max(last_end,position+duration)
    head=['ustx_version: "0.10"','name: %s'%_quote(name),'comment: ""','output_dir: Vocal','cache_dir: UCache',
          'key: 0','time_signatures:','- bar_position: 0','  beat_per_bar: 4','  beat_unit: 4',
          'tempos:','- position: 0','  bpm: %g'%bpm,
          'tracks:','- singer: ""','  track_name: %s'%_quote('Vocal'),'  track_color: Blue','  mute: false',
          '  solo: false','  volume: 0','  pan: 0','  voice_color_names: [""]','  track_expressions: []',
          'voice_parts:','- name: Part1','  comment: ""','  track_no: 0','  position: 0',
          '  duration: %d'%max(last_end,PPQ)]
    body=['  notes:']+blocks if blocks else ['  notes: []']
    xs,ys=_pitd(samples,ordered,hop,origin,bpm)
    tail=['  curves:','  - abbr: pitd','    xs: %s'%xs,'    ys: %s'%ys] if xs else ['  curves: []']
    tail+=['  masked_curves: []','  wave_parts: []']
    return '\n'.join(head+body+tail)+'\n'
