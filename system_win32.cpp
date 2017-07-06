
#include "system.hpp"

#include "libs/stack_walker_win32.hpp"

namespace jup {

void win_last_errmsg() {
    auto err = GetLastError();
    char* msg = nullptr;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        nullptr,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&msg,
        0,
        nullptr
    );
    err_msg(msg, err);
}

class MyStackWalker : public StackWalker {
    using StackWalker::StackWalker;
protected:
    void OnSymInit(LPCSTR szSearchPath, DWORD symOptions, LPCSTR szUserName) override {}
    void OnLoadModule(LPCSTR img, LPCSTR mod, DWORD64 baseAddr, DWORD size, DWORD result,
        LPCSTR symType, LPCSTR pdbName, ULONGLONG fileVersion) override {}
    
    void OnCallstackEntry(CallstackEntryType eType, CallstackEntry& entry) override {
        if (eType == lastEntry || entry.offset == 0) return;
        CHAR buffer[STACKWALK_MAX_NAMELEN];

        if (std::strcmp(entry.name, "ShowCallstack") == 0) return;
        if (std::strcmp(entry.name, "_assert_fail")  == 0) return;
        if (std::strcmp(entry.name, "die")  == 0) return;
        
        if (entry.name[0] == 0)
            strcpy_s(entry.name, "(function-name not available)");
        if (entry.undName[0] != 0)
            strcpy_s(entry.name, entry.undName);
        if (entry.undFullName[0] != 0)
            strcpy_s(entry.name, entry.undFullName);
        if (entry.lineFileName[0] == 0) {
            strcpy_s(entry.lineFileName, "(filename not available)");
            if (entry.moduleName[0] == 0)
                strcpy_s(entry.moduleName, "(module-name not available)");
            _snprintf_s(buffer, STACKWALK_MAX_NAMELEN, "%p (%s): %s: %s\n", (LPVOID)entry.offset,
                entry.moduleName, entry.lineFileName, entry.name);
        } else {
            std::snprintf(buffer, sizeof(buffer), "%s:%d: %s\n",
                entry.lineFileName, (int)entry.lineNumber, entry.name);
        }
        OnOutput(buffer);
    }
    
    void OnOutput(LPCSTR szText) override {
        jerr << "  " << szText;
    }
    void OnDbgHelpErr(LPCSTR szFuncName, DWORD gle, DWORD64 addr) override {
        //char buf[256];
        //std::snprintf(buf, sizeof(buf), "%p", (void const*)addr);
        //jerr << "Error: " << szFuncName << " at " << buf << '\n';
        //win_last_errmsg();
    }
};

void die() {
    jerr << "\nStack trace:\n";
    MyStackWalker sw;
    sw.ShowCallstack();
    
    // This would be the more proper way, but I can't get mingw to link a recent
    // version of msvcr without recompiling a lot of stuff.
    //_set_abort_behavior(0, _WRITE_ABORT_MSG);
    CloseHandle(GetStdHandle(STD_ERROR_HANDLE));
    
    std::abort();
}

int get_terminal_width() {
    CONSOLE_SCREEN_BUFFER_INFO info;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
    int width = info.srWindow.Right - info.srWindow.Left + 1;
    
    // This does not always work, make 80 minimum as a workaround
    if (width == 1) width = 80;
    return width;
}

} /* end of namespace jup */

