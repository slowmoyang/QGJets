PROGS := full_steps
all : $(PROGS)

% : %.cc
	g++ -g `root-config --cflags --libs` $< -o $@
