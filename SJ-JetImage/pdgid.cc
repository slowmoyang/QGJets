#include <memory>

enum class PdgId: int {
    kElectron = 11,
    kPositron = -11,
    kAntiElectron = -11,
    kMuon = 13,
    kAntiMuon = -13,
    kPhoton = 22,
};

bool operator== (int pid0, PdgId pid1)
{
    return pid0 == static_cast<int>(pid1);
}
