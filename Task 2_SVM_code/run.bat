@echo off

call C:\Users\12053\anaconda3\Scripts\activate.bat C:\Users\12053\anaconda3

call conda activate test

for %%l in (0, 1, 2) do (
	for %%C in (0.0001, 0.0005, 0.0008, 0.001, 0.003, 0.005, 0.01) do (
		call python SVM.py -C %%C -T 10000 -l %%l
	)
	for %%T in (5000, 8000, 10000, 12000, 15000) do (
		call python SVM.py -C 0.001 -T %%T -l %%l
	)
)
pause

