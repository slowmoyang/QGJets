PROGS := merge_and_shuffle
PROG=merge_and_shuffle
SRC_FILES=$(wildcard ../src/*.cc)
OBJ_FILES=$(SRC_FILES:../src/%.cc=../obj/%.o)
all : $(PROG)

../obj/%.o : ../src/%.cc
	g++ -c $< -o $@ -g `root-config --cflags`

$(PROG) : $(PROG).cc $(OBJ_FILES)
	g++ $^ -o $@ -g `root-config --cflags --libs`
