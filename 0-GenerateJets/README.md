# Running MG5

Use the shell scripts in MG5/ (run from the MG5 directory)

# Running Sherpa

You may need to 
`unset SHERPA_LIBRARY_PATH`
`unset SHERPA_INCLUDE_PATH`
`unset SHERPA_SHARE_PATH`
if you setup cmsenv

In the Sherpa/ directory run

`singularity run ~iwatson/Images/Sherpa.img -f pp_qq.input`
Occasionally, Sherpa needs to compile some libs it creates before generating MC, in that case also run
`./makelibs`
`singularity run ~iwatson/Images/Sherpa.img -f pp_qq.input`
then run Delphes
`singularity exec ~iwatson/Images/Madgraph.img /code/MG5_aMC_v2_6_0/Delphes/DelphesHepMC ../Cards/delphes_card_CMS.tcl ../Sherpa_pp_qq.root pp_qq.hepmc2g`

# Running Herwig

(May need to unset Herwig variables if this fails)
In the Herwig/ directory run

`singularity run ~iwatson/Images/Herwig.img build pp_gg.in`
`singularity run ~iwatson/Images/Herwig.img run pp_gg.run`
`singularity exec ~iwatson/Images/Madgraph.img /code/MG5_aMC_v2_6_0/Delphes/DelphesHepMC ../Cards/delphes_card_CMS.tcl ../Herwig_pp_gg.root pp_gg.hepmc`
