Whale Coda Information Structure

Code + data to reproduce information-theoretic analyses of sperm whale codas
using a rhythm×tempo unit and contextual overlays (rubato, ornamentation).

Key numbers
-----------------------
Codas: 3,673 | Rhythms: 18 | Tempo types: 5
H(Unit): 3.490 bits
H(Unit) [rubato subset]: 2.962 bits
H(Unit|rubato): 2.889  -> I(Unit;rubato)=0.073 bits
H(Unit|ornament): 3.452 -> I(Unit;orn)=0.038 bits
H(Unit|rubato,orn): 2.861 -> I(Unit;rub,orn)=0.101 bits
Entropy rate (order-1, observed): 2.079 bits
Null (within-exchange shuffles): mean 2.549; 95% [2.517, 2.581]
Mean ornament-free duration: 0.912 s -> 3.82 bits/s (marginal)

Included
--------------
- Quantifying_Information_Structure_in_Sperm_Whale_Codas.pdf (final report)
- data/sperm-whale-dialogues.csv
- data/mean_codas.p               (rhythm templates)
- data/ornaments.p                (ornament annotations)
- run_dialogues.py
- src/assign_rhythm_by_means.py
- src/sw_loader.py
- src/entropies.py
- src/bootstrap.py                (Not used for analysis)
- requirements.txt 
- LICENSE-DATA.txt                (data credit & license)

