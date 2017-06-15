
# Generic C++ Makefile

# cv2pdb is required to build this project. You can use 'make init' to install a prebuild binary.

TARGET = schaf
LIBS = -lmsvcr100 -lWs2_32 -lversion -static-libstdc++ -static-libgcc
CXX = g++
CXXFLAGS = -g -Wall -Werror -pedantic -fmax-errors=2
CPPFLAGS = -D__MSVCRT_VERSION__=0x1000 -std=c++1y
LDFLAGS  = -Wall
EXEEXT = .exe
CV2PDB = cv2pdb
TMPDIR = build_files
PRE_HEADER = $(TMPDIR)/global.hpp.gch

.PHONY: default all clean test init
.SUFFIXES:

all: default

SOURCES = $(wildcard *.c) $(wildcard *.cpp)
OBJECTS = $(SOURCES:%.cpp=$(TMPDIR)/%.o)
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
DEPS    = $(SOURCES:%.cpp=$(TMPDIR)/%.d)

$(PRE_HEADER): global.hpp
	@mkdir -p $(TMPDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(TMPDIR)/%.d: %.cpp $(HEADERS)
	@mkdir -p $(TMPDIR)
	@set -e; $(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,$(TMPDIR)/\1.o $@ : ,g' > $@;

$(TMPDIR)/%.o: %.cpp $(PRE_HEADER)
	@mkdir -p $(TMPDIR)
	$(CXX) $(CPPFLAGS) -I $(TMPDIR) -include global.hpp $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

default: $(TARGET)

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TMPDIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

$(TARGET): $(TMPDIR)/$(TARGET)
	$(CV2PDB) $<$(EXEEXT) $@$(EXEEXT)

init:
	mkdir -p /usr/local/bin
	wget https://ci.appveyor.com/api/projects/rainers/visuald/artifacts/cv2pdb.exe?job=Environment%\
	3A%20os%3DVisual%20Studio%202013%2C%20VS%3D12%2C%20APPVEYOR_BUILD_WORKER_IMAGE%3DVisual%20Studi\
	o%202015 -O /usr/local/bin/cv2pdb.exe
	cp -f eer.py /usr/local/bin/eer

clean:
	-rm -f *.o *.d *~
	-rm -f $(TARGET)$(EXEEXT)
	-rm -f $(TMPDIR)/*
	-rmdir $(TMPDIR)
