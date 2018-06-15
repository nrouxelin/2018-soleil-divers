# Description du bug
Erreur MPI lorsque le maillage est trop fin (`carre_non_unif.ini`) ou quand le cas à traiter est trop gros (`galbrun_soleil.ini`).
Dans ces cas on exécute le programme sur beaucoup de processus (~ 20).

# Erreur obtenue
```
Fatal error in PMPI_Test: Invalid MPI_Request, error stack:                                                                                   │
PMPI_Test(183): MPI_Test(request=0x9d08f98, flag=0x7ffc7e2f0e70, status=0x7ffc7e2f0ea0) failed                                                │
PMPI_Test(132): Invalid MPI_Request
```
Cette erreur apparait lors de l'étape `Factorization step`

