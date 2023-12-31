TOPDIR ?= ./
CROSS_COMPILE:=mips-linux-gnu-

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip
libtype ?= muclibc
build_type ?= release
#build_type ?= profile
#build_type ?= debug
#build_type ?= nmem

CXXFLAGS := -std=c++11 -mfp64 -mnan=2008 -mabs=2008 -Wall -EL -O3 -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm
INCLUDES := -I$(TOPDIR)/include

ifeq ($(libtype), muclibc)
	CXXFLAGS += -muclibc
endif

ifeq ($(build_type), release)
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc -lvenus -laip -ldrivers
    else
        LIBS := -L$(TOPDIR)/lib/glibc -lvenus -laip -ldrivers
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_yolov5s_bin_uclibc_release
    else
        TARGET = venus_yolov5s_bin_glibc_release
    endif

else ifeq ($(build_type), profile)
    CXXFLAGS += -DVENUS_PROFILE
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus.p -laip -ldrivers
    else
        LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus.p -laip -ldrivers
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_yolov5s_bin_uclibc_profile
    else
        TARGET = venus_yolov5s_bin_glibc_prolfile
    endif

else ifeq ($(build_type), debug)
    CXXFLAGS += -DVENUS_DEBUG
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/  -lvenus.d -laip -ldrivers
    else
        LIBS := -L$(TOPDIR)/lib/glibc/  -lvenus.d -laip -ldrivers
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_yolov5s_bin_uclibc_debug
    else
        TARGET = venus_yolov5s_bin_glibc_debug
    endif

else ifeq ($(build_type), nmem)
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus -laip -ldrivers.m
    else
        LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus -laip -ldrivers.m
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_yolov5s_bin_uclibc_nmem
    else
        TARGET = venus_yolov5s_bin_glibc_nmem
    endif

endif

OBJS := inference.o
#OBJS := inference_nv12.o

%.o:%.cpp
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean: 
	rm -f $(TARGET) $(OBJS)
