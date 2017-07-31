#pragma once

namespace jup {

struct Network_state;

Network_state* network_init();
void network_free(Network_state* state);
void network_main();

} /* end of namespace jup */
