CC := gcc
LD := gcc
LDFLAGS := -fPIC
CFLAGS := -fPIC

WARN := -Wall -Wextra
GDB := -ggdb3
ASAN := -fsanitize=address
#UBSAN := -fsanitize=undefined

ifeq ($(DEBUG),1)
OPT :=
CFLAGS += $(ASAN)
LDFLAGS += $(ASAN)
CFLAGS += $(UBSAN)
LDFLAGS += $(UBSAN)
else
OPT := -Ofast
#OPT += -march=native
endif

CFLAGS += $(WARN) $(GDB) $(OPT)

LIB := libspline.so
PROG := spline

.PHONY: all
all: $(PROG) $(LIB)

.PHONY: clean
clean:
	$(RM) $(PROG) $(LIB) spline.o

.PHONY: check
check: $(PROG)
	./$(PROG)

.PHONY: check_big
check_big: python_fit.py $(LIB)
	python $<

$(LIB): LDFLAGS += -shared
$(LIB): spline.o
	$(LINK.o) $(OUTPUT_OPTION) $^

$(PROG): spline.o
	$(LINK.o) $(OUTPUT_OPTION) $^

spline.o: spline.h
