shots=[163303]
include_ece=True
include_bes=True
bes_s_or_f='s' # s for slow (10kHz sampling), f for fast (10MHz sampling)
inculde_co2=True

pipeline = Pipieline(shots)

if include_ece:
    signal=MdsSignal('.PROFS.ECEPROF',
                     'ECE',
                     location='remote://atlas.gat.com',
                     dims=['times','channels'])
    pipeline.fetch('ece_full',signal)

if include_bes:
    for i in range(1,65):
        signal=PtDataSignal('bes{}u{:02d}'.format(bes_s_or_f,i))
        pipeline.fetch('bes_{}_{}'.format(bes_s_or_f,i),signal)

# see https://diii-d.gat.com/diii-d/Mci for details 
if include_co2:
    channels=['v1','v2','v3','r0']
    names=['den', 'stat'] # line-integrated density, status
    for channel in channels:
        signal=MdsSignal('den{}'.format(channel),'BCI',location='remote://atlas.gat.com')
        pipeline.fetch('den{}'.format(channel),signal)
        signal=MdsSignal('stat{}'.format(channel),'BCI',location='remote://atlas.gat.com')
        pipeline.fetch('stat{}'.format(channel),signal)
        signal=PtDataSignal('stat{}'.format(channel))
        pipeline.fetch('stat{}'.format(channel),signal)
