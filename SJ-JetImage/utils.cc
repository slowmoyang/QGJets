#include <vector>

#include "TString.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TList.h"
#include "TSystemFile.h"
#include "TCollection.h"




std::vector< TString > ListDir(TString path,
                               TString extension="",
                               TString pattern="",
                               bool without_dir=true){

    // TSystemDirectory(const char* dirname, const char* path)
    TSystemDirectory dir(path, path);
    TList *entries = dir.GetListOfFiles();
    TSystemFile *entry;
    TString entry_name;
    TIter iter(entries);

    std::vector< TString > entries_path;

    while ( ( entry = (TSystemFile*) iter() ) ) {
        if(without_dir and entry->IsDirectory())
            continue;

        entry_name = entry->GetName();

        if(not entry_name.EndsWith(extension))
            continue;

        if(not entry_name.Contains(pattern))
            continue;

        TString entry_path = gSystem->ConcatFileName(path, entry_name.Data());

        entries_path.push_back(entry_path);
    }

    return entries_path;
}


