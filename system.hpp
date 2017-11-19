#pragma once

#include "buffer.hpp"

namespace jup {

double elapsed_time();
void init_elapsed_time(double val = 0);

int get_terminal_width();

void init_utf8();
bool is_enabled_utf8();

jup_str get_error_msg_system(jup_str code);

} /* end of namespace jup */
