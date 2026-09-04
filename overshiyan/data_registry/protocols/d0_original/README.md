# D0 original-label validation protocol

This protocol contains the original labels for the same fixed 897 validation
images used by Data3. Files were mapped from the organizer-style names to Data3's
sequential stems using:

`shiyan/data2/manifests/rename_0001_4481_manifest.csv`

The frozen mapping contains 897 labels and 4243 objects. Every copied label hash
matches its source hash; see `mapping_manifest.csv`. These labels are evaluation
evidence only. They are never included in Data3 training or hard-negative mining.

The protocol exists because manual annotation changes can improve one label policy
while moving away from the hidden platform policy. Identical predictions are scored
against D3 and D0, and candidate selection uses the worse result.
