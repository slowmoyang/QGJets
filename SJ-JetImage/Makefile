PROGS := full_steps SummedJetImage
all : $(PROGS)

% : %.cc
	g++ -g `root-config --cflags --libs` $< -o $@
