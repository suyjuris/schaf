
# Generic C++ Makefile

# cv2pdb is required to build this project. You can use 'make init' to install a prebuild binary.

TARGET = schaf
LIBS = -lmsvcr100 -lWs2_32 -lversion -lz -static-libstdc++ -static-libgcc
CXX = g++
CXXFLAGS = -g -Wall -Werror -pedantic -fmax-errors=2
CPPFLAGS = -D__MSVCRT_VERSION__=0x1000 -std=c++1y
LDFLAGS  = -Wall
EXEEXT = .exe
CV2PDB = cv2pdb
LIBDIR = libs
TMPDIR = build_files
PRE_HEADER = $(TMPDIR)/global.hpp.gch

.PHONY: default all clean test init
.SUFFIXES:

all: default

SOURCES = $(wildcard *.cpp) $(wildcard $(LIBDIR)/*.cpp)
OBJECTS = $(SOURCES:%.cpp=$(TMPDIR)/%.o)
HEADERS = $(wildcard *.hpp) $(wildcard $(LIBDIR)/*.hpp)
DEPS    = $(SOURCES:%.cpp=$(TMPDIR)/%.d)

test:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Headers: $(HEADERS)"
	@echo "Deps:    $(DEPS)"

$(PRE_HEADER): global.hpp
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(TMPDIR)/%.d: %.cpp $(HEADERS)
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
	@set -e; $(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,$(TMPDIR)/\1.o $@ : ,g' > $@;

$(TMPDIR)/%.o: %.cpp $(PRE_HEADER)
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
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
	-rm -f *~ $(TARGET).pdb $(TARGET)$(EXEEXT) $(TMPDIR)/*.* $(TMPDIR)/$(LIBDIR)/*.*
	-rmdir $(TMPDIR)/$(LIBDIR) $(TMPDIR)
