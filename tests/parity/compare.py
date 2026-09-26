import json, re, sys
nat=[json.loads(l) for l in open(sys.argv[1])]
rs=[json.loads(l) for l in open(sys.argv[2])]
show=int(sys.argv[3]) if len(sys.argv)>3 else 0
strip=lambda s: re.sub(r',"(prefill_tps|decode_tps|peak_ram_mb)":[0-9.]+','',s or '')
k=lambda c: json.dumps(c,sort_keys=True)
same_raw=same_final=dec=0; gate=[]; conf=[]
for n,r in zip(nat,rs):
    a,b=strip(n['raw']),strip(r['raw'])
    same_raw += a==b
    same_final += k({x:y for x,y in n['final'].items() if x not in('prefill_tps','decode_tps','peak_ram_mb')})==k({x:y for x,y in r['final'].items() if x not in('prefill_tps','decode_tps','peak_ram_mb')})
    if a==b: continue
    na,ra=json.loads(n['raw'] or 'null'),json.loads(r['raw'] or 'null')
    if na is None or ra is None: gate.append(('NORAW',n['query'])); continue
    nc=na['function_calls']+na['suppressed_calls']; rc=ra['function_calls']+ra['suppressed_calls']
    if k(nc)==k(rc) and na['reasoning']==ra['reasoning']:
        dec+=1
        if k(na['function_calls'])!=k(ra['function_calls']) or k(na.get('validation','ABSENT'))!=k(ra.get('validation','ABSENT')) or na['type']!=ra['type']:
            gate.append((n['query'][:55], 'N', json.dumps(na['function_calls'])[:90], len(na['suppressed_calls']), json.dumps(na.get('validation','ABSENT')), 'R', json.dumps(ra['function_calls'])[:90], len(ra['suppressed_calls']), json.dumps(ra.get('validation','ABSENT'))))
        elif na['confidence']!=ra['confidence']: conf.append((na['confidence'],ra['confidence'],n['query'][:50]))
print(f"raw identical {same_raw}/{len(nat)}  final identical {same_final}/{len(nat)}  same decode but raw differs {dec}: gate {len(gate)} conf {len(conf)}")
for g in gate[:show]: print(g)
for c in conf[:show]: print(c)
