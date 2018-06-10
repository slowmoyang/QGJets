ALL=analysis analysis_lite analysis_yang

all : $(ALL)

analysis : analysis.cxx
	g++ -DWOOJIN_VARIABLES -g -std=c++1y -o analysis analysis.cxx -L/code/MG5_aMC_v2_6_0/Delphes/ -I../../install/include `root-config --cflags --libs` -I/code/MG5_aMC_v2_6_0/Delphes/ -I/code/MG5_aMC_v2_6_0/Delphes/external -lEG -lDelphes

analysis_lite : analysis.cxx
	g++ -g -std=c++1y -o analysis_lite analysis.cxx -L/code/MG5_aMC_v2_6_0/Delphes/ -I../../install/include `root-config --cflags --libs` -I/code/MG5_aMC_v2_6_0/Delphes/ -I/code/MG5_aMC_v2_6_0/Delphes/external -lEG -lDelphes

analysis_yang : analysis_yang.cxx
	g++ -DWOOJIN_VARIABLES -g -std=c++1y -o analysis_yang analysis_yang.cxx -L/code/MG5_aMC_v2_6_0/Delphes/ -I../../install/include `root-config --cflags --libs` -I/code/MG5_aMC_v2_6_0/Delphes/ -I/code/MG5_aMC_v2_6_0/Delphes/external -lEG -lDelphes

clean :
	rm -rf $(ALL)
