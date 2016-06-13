CC = g++
CXX = g++
#CXXFLAGS = `pkg-config --cflags opencv`
#LDFLAGS = `pkg-config --libs opencv`
CFLAGS = -Wall -I./wrapplot -I./wrapunix -g
#OPENCV_INC = -I/usr/local/include/opencv2
OPT_FLAG = -O3 -mtune=i686
#CPPFLAGS = -g $(OPT_FLAG) $(OPENCV_INC) $(IPL_INC) $(CXXFLAGS) $(LDFLAGS)
CPPFLAGS = -g $(OPT_FLAG) $(CXXFLAGS) $(LDFLAGS)
LDLIBS = $(IPL_LIB) $(GPG_LIB) -lm -lgsl -lgslcblas 

all:	$(TARGET)

clean: 
	rm -f *.o $(TARGETS)



