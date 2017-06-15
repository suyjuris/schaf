

using namespace jup;

#if 0

void print_usage(c_str argv0) {
    using namespace cmd_options;
	jerr << "Usage:\n  " << argv0 << " [options]\n  " << argv0 << " --help\n\n"
		<< "Options:\n"
		<< " " << MASSIM_LOC << " [path]  The location of the massim server, used to start the int"
		<< "ernal server wrapper. (Should contain a server/ subdirectory.)\n"
		<< " " << CONFIG_LOC << " [path]  The location of a specific configuration to use for star"
		<< "ting the internal server (there is a default value).\n"
		<< " " << HOST_IP << " [ip]    The IP address for connecting with an external server.\n"
		<< " " << HOST_PORT << " [port]  The port for connecting with an external server.\n"
		<< " " << DUMP_XML << " [path]  Debug option. If this is specified all xml messages betwe"
		<< "en the server and the program are dumped into a file.\n\n"
		<< " " << ADD_AGENT << " [name] [password]  The login credentials for an agent. This opti"
		<< "on may be specified multiple times. It also may use the % symbol at the end of a name,"
		<< "which will be replaced by the numbers 1 to " << agents_per_team << ". For compatibilit"
		<< "y the , symbol has the same effect.\n"
		<< " " << ADD_DUMMY << " [name] [password]  Like " << ADD_AGENT << " but adds a dummy tha"
		<< "t does not do anything.\n"
		<< " " << LOAD_CFGFILE << " [path]  The file is interpreted as a configfile. See below for"
		<< " the syntax.\n\n"
		<< " The programm determines automatically whether to run the internal server or connect t"
		<< "o an external server by checking with options have been specified (" << MASSIM_LOC
		<< " and " << CONFIG_LOC << " respectively, the latter has higher priority).\n\n"
		<< "Configfile syntax:\n"
		<< " The configfile is split into lines. Each lines either starts with an '#', which cause"
		<< "s it to be ignored, or has the following form:\n   option arg1 [arg2]\n\n "
        << LAMPE_SHIP << " [ship]  Specifies the type of operation. Must be one of:\n    test  To "
        << "run a quick self-check\n    stats  To collect statistical information about simulation"
        << "s and append them to the specified file\n\n";
}

/**
 * Parse the command-line arguments
 */
bool parse_cmdline(int argc, c_str const* argv, Server_options* into, bool no_recursion = false) {
    using namespace cmd_options;

    int i;
    auto pop = [&i, argc, argv](Buffer_view* into) {
        if (++i < argc) {
            *into = argv[i];
            return true;
        } else {
            jerr << "Error: while parsing arguments: unexpected end of arguments\n";
            return false;
        }
    };
    
    for (i = 0; i < argc; ++i) {
        Buffer_view arg {argv[i]};
        if (!no_recursion and (arg == "-h" or arg == "--help" or arg == "-?" or arg == "/?")) {
            print_usage(argv[-1]);
            std::exit(1);
        } else if (arg == MASSIM_LOC) {
            into->use_internal_server = true;
            if (not pop(&into->massim_loc)) {
                return false;
            }
        } else if (arg == CONFIG_LOC) {
            if (not pop(&into->config_loc)) {
                return false;
            }
        } else if (arg == HOST_IP) {
            into->use_internal_server = false;
            if (not pop(&into->host_ip)) {
                return false;
            }
        } else if (arg == HOST_PORT) {
            if (not pop(&into->host_port)) {
                return false;
            }
        } else if (arg == DUMP_XML) {
            if (not pop(&into->dump_xml)) {
                return false;
            }
        } else if (arg == STATS_FILE) {
            if (not pop(&into->statistics_file)) {
                return false;
            }
        } else if(arg == LAMPE_SHIP) {
            Buffer_view tmp;
            if (not pop(&tmp)) {
                return false;
            }
			if (tmp == LAMPE_SHIP_TEST) {
				into->ship = Server_options::SHIP_TEST;
			} else if (tmp == LAMPE_SHIP_STATS) {
				into->ship = Server_options::SHIP_STATS;
			} else if (tmp == LAMPE_SHIP_PLAY) {
				into->ship = Server_options::SHIP_PLAY;
			} else {
				jerr << "Error: unknown ship '" << tmp << "', must be one of " << LAMPE_SHIP_TEST
                     << ", " << LAMPE_SHIP_STATS << " or " << LAMPE_SHIP_PLAY << '\n';
				return false;
            }
        } else if (arg == ADD_AGENT or arg == ADD_DUMMY) {
            bool is_dumb = arg == ADD_DUMMY;
            Buffer_view name, password;
            if (not pop(&name)) {
                return false;
            }
            
            if (not pop(&password)) {
                jerr << "Note: " << ADD_AGENT << " expects both the agent name and the password as "
                     << "separate arguments.\n";
                return false;
            }

            if (name.end()[-1] == '%' or name.end()[-1] == ',') {
                for (int i = 1; i <= agents_per_team; ++i) {
                    constexpr int space = 8;
                    int name_off = into->_string_storage.size();
                    into->_string_storage.append(name);
                    into->_string_storage.reserve_space(8);
                    int written = std::snprintf(into->_string_storage.end() - 1, space, "%d", i);
                    assert(written < space);
                    into->_string_storage.addsize(written);
            
                    into->agents.emplace_back();
                    into->agents.back().name = Buffer_view {
                        into->_string_storage.data() + name_off,
                        into->_string_storage.size() - name_off - 1
                    };
                    into->agents.back().password = password;
                    into->agents.back().is_dumb = is_dumb;
                }
            } else {
                into->agents.emplace_back();
                into->agents.back().name = name;
                into->agents.back().password = password;
                into->agents.back().is_dumb = is_dumb;
            }
		} else if (arg == LOAD_CFGFILE) {
			if (no_recursion) {
				jerr << "Error: tried to read a configfile while reading a configfile.\n";
				return false;
			}
			Buffer_view path;
			if (not pop(&path)) {
				return false;
			} else if (not file_exists(path)) {
				jerr << "Error: Could not load configfile: File does not exist.\n You specified the"
					<< "file:\n  " << path.c_str() << '\n';
				return false;
			}

			// Read and parse the configfile
			std::vector<c_str> args;
			std::ifstream is{ path.c_str() };
			if (not is) {
				jerr << "Error: Invalid stream while reading configfile.\n";
				return false;
			}
			int begin_file = into->_string_storage.size();
			constexpr int space = 1024;
			do {
				into->_string_storage.reserve_space(space);
				is.read(into->_string_storage.end(), into->_string_storage.space());
				into->_string_storage.addsize(is.gcount());
			} while (into->_string_storage.space() == 0);
			into->_string_storage.append("", 1);

			int state = 0;
			int last = begin_file;
			for (int i = begin_file; i < into->_string_storage.size(); ++i) {
				if (state == 0 or state == 4) {
					if (into->_string_storage[i] == ' ' or into->_string_storage[i] == '\t') {
						if (last < i) {
							into->_string_storage[i] = '\0';
							args.push_back(&into->_string_storage[last]);
							// Handle these options differently, because they need two arguments
							if (std::strcmp(args.back(), ADD_AGENT) == 0 and state == 0) {
								state = 4;
							} else if (std::strcmp(args.back(), ADD_DUMMY) == 0 and state == 0) {
								state = 4;
							} else {
								state = 1;
							}
						}
						last = i + 1;
					} else if (into->_string_storage[i] == '\n') {
						last = i + 1;
					} else if (into->_string_storage[i] == '#') {
						state = 2;
					}
				} else if (state == 1) {
					if (into->_string_storage[i] == '\n') {
						state = 0;
						args.push_back(&into->_string_storage[last]);
						into->_string_storage[i] = '\0';
						last = i + 1;
					}
				} else if (state == 2) {
					if (into->_string_storage[i] == '\n') {
						last = i + 1;
						state = 0;
					}
				} else {
					assert(false);
				}
			}

			if (not parse_cmdline(args.size(), args.data(), into, true)) {
				jerr << "Error: ... while parsing configfile. The file is located at:\n  "
					 << path.c_str() << '\n';
				return false;
			}
		} else {
            jerr << "Error: Invalid option. The option was:\n  " << arg.c_str() << '\n';
            return false;
        }
        
    }
    return true;
}

#endif

int main(int argc, c_str const* argv) {
    jerr << argv[0] << '\n';
    assert(false);
	return 0;
}
