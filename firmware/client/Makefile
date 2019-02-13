Configuration?=Release
VPATH = x64/$(Configuration)

SRCS  = \
	load_firmware.cpp \
	bond_request.cpp \
	bond_response.cpp \
	brainslice_parameters.cpp \
	$(wildcard x64/$(Configuration)/*_types.cpp) \
	$(wildcard x64/$(Configuration)/*_client.cpp)

SRCFILES = $(notdir $(SRCS))
OBJFILES = $(SRCFILES:.cpp=.o)
OBJS = $(addprefix x64/$(Configuration)/, $(OBJFILES))

WARNINGS=-Wswitch -Wno-deprecated-declarations -Wempty-body -Wreturn-type
WARNINGS:=$(WARNINGS) -Wno-format -Wuninitialized
WARNINGS:=$(WARNINGS) -Wunreachable-code -Wunused-function -Wunused-value
WARNINGS:=$(WARNINGS) -Wall -Werror
WARNINGS:=$(WARNINGS) -Wno-unknown-pragmas -Wno-reorder -Wno-unused-variable -Wno-parentheses -Wno-sign-compare

CFLAGS:=$(CFLAGS) -fpic $(WARNINGS) -std=c++14 -fno-strict-aliasing
CFLAGS:=$(CFLAGS) -g2 -gdwarf-2 -O3 -fthreadsafe-statics
CFLAGS:=$(CFLAGS) -fno-omit-frame-pointer -fexceptions
CFLAGS:=$(CFLAGS) -march=native

CDEFINES:=-D CLIENT_DLL

INC += -I./x64/$(Configuration)
INC += -I$(DevKit)/software/inc
INC += -I$(DevKit)/software/inc/3rdparty
INC += -I$(PkgBond_Cpp)/lib/native/include

LIB_FLAGS += -Wl,--no-undefined

LIBS += -lstdc++
LIBS += -lm

all: x64/$(Configuration)/libclient.so

x64/$(Configuration)/libclient.so: $(OBJS)
	$(CC) $(LIB_FLAGS) -shared -o $@ $^ $(LIBS)

clean:
	rm -rf x64/$(Configuration)

x64/$(Configuration)/%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) $(CDEFINES) -c $< -o $@
