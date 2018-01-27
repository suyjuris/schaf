
# Generic C++ Makefile

#TODO Add Support for Windows again

ifeq ($(OS),Windows_NT)
  MODE = WINDOWS
else
  MODE = LINUX
endif

TARGET   = schaf
CXX      = g++
CXXFLAGS = -g -Wall -Werror -pedantic -fmax-errors=2
CPPFLAGS = -std=c++1z -DJUP_OS=$(MODE) -DJUP_OS_$(MODE) -Ibuild_files/include_linux
LDFLAGS  = -Wall
LIBS     = -lz -ltensorflow_cc -ltensorflow_framework -lstdc++fs -lpthread

# Put your custom configuration in there
make.config:
	echo '# Custom build configuration in here' > $@
-include make.config

ifeq ($(SCHAF_FAST),1)
  CXXFLAGS += -O3 -march=native
  CPPFLAGS += -DNDEBUG
else
  CXXFLAGS += -O0
endif

ifeq ($(MODE),WINDOWS)
  LIBS += -lWs2_32 -lversion -static-libstdc++ -static-libgcc
else
  LDFLAGS += -rdynamic
endif

ifeq ($(USE_PROFILER),1)
  LIBS += -lprofiler
  CPPFLAGS += -DJUP_USE_PROFILER
endif

ifeq ($(MODE),WINDOWS)
  EXEEXT = .exe
  CV2PDB = cv2pdb
else
  EXEEXT =
endif

ifeq ($(SCHAF_FAST),1)
  SUFFIX = _fast
else
  SUFFIX =
endif

LIBDIR = libs
TMPDIR = build_files
PRE_HEADER = $(TMPDIR)/global$(SUFFIX).hpp.gch
TARGET_NAME = $(TARGET)$(SUFFIX)$(EXEEXT)

TMPFILES = *~ $(TARGET_NAME)
ifeq ($(MODE),WINDOWS)
  TMPFILES += $(TARGET)$(SUFFIX).pdb
endif

.PHONY: default all clean print_config init distclean
.SUFFIXES:
.DEFAULT_GOAL := default

all: default

SOURCES = $(wildcard *.cpp) $(wildcard $(LIBDIR)/*.cpp)
HEADERS = $(wildcard *.hpp) $(wildcard $(LIBDIR)/*.hpp)

ifeq ($(MODE),WINDOWS)
  SOURCES := $(filter-out %_linux.cpp,$(SOURCES))
  HEADERS := $(filter-out %_linux.hpp,$(HEADERS))
else
  SOURCES := $(filter-out %_win32.cpp,$(SOURCES))
  HEADERS := $(filter-out %_win32.hpp,$(HEADERS))
endif

OBJECTS = $(SOURCES:%.cpp=$(TMPDIR)/%$(SUFFIX).o)
DEPS    = $(SOURCES:%.cpp=$(TMPDIR)/%$(SUFFIX).d)

print_config:
	@echo "Mode:     $(MODE)"
	@echo "Sources:  $(SOURCES)"
	@echo "Objects:  $(OBJECTS)"
	@echo "Headers:  $(HEADERS)"
	@echo "Deps:     $(DEPS)"
	@echo "CPPFLAGS: $(CPPFLAGS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS:  $(LDFLAGS)"

init:
	mkdir -p $(TMPDIR)/include_linux
	tar -xjf $(LIBDIR)/tensorflow_linux.tar.bz2 -C $(TMPDIR)/include_linux
ifeq ($(MODE),WINDOWS)
	mkdir -p /usr/local/bin
	wget https://ci.appveyor.com/api/projects/rainers/visuald/artifacts/cv2pdb.exe?job=Environment%\
	3A%20os%3DVisual%20Studio%202013%2C%20VS%3D12%2C%20APPVEYOR_BUILD_WORKER_IMAGE%3DVisual%20Studi\
	o%202015 -O /usr/local/bin/cv2pdb.exe
endif

$(PRE_HEADER): global.hpp
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(TMPDIR)/%$(SUFFIX).d: %.cpp $(HEADERS)
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
	@set -e; $(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,$(TMPDIR)/\1$(SUFFIX).o $@ : ,g' > $@;

$(TMPDIR)/%$(SUFFIX).o: %.cpp $(PRE_HEADER)
	@mkdir -p $(TMPDIR) $(TMPDIR)/$(LIBDIR)
	$(CXX) $(CPPFLAGS) -I$(TMPDIR) -include global.hpp $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

default: $(TARGET_NAME)

.PRECIOUS: $(TARGET_NAME) $(OBJECTS)

ifeq ($(MODE),WINDOWS)
$(TMPDIR)/$(TARGET_NAME): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

$(TARGET_NAME): $(TMPDIR)/$(TARGET_NAME)
	$(CV2PDB) -C $< $@
else
$(TARGET_NAME): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@
endif

clean:
	-rm -f $(TMPFILES) $(TMPDIR)/*.* $(TMPDIR)/$(LIBDIR)/*.*

distclean:
	test -n "$(TMPDIR)"
	-rm -rf "./$(TMPDIR)"
