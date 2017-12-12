PROGS := step1_selection_and_labeling step2_first_merge step3_random_shuffle step4_second_merge step5_ttsplit jet_image_maker

all : $(PROGS)

% : %.cc
	g++ -g `root-config --cflags --libs` $< -o $@
