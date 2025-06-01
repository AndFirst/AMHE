# Projekt AMHE

Projekt zrealizowany w ramach przedmiotu **AMHE**, dotyczący porównania skuteczności algorytmów optymalizacyjnych opartych na strategii ewolucyjnej z adaptacyjną macierzą kowariancji.

## Opis

Celem projektu było zaimplementowanie i przetestowanie wariantów algorytmu MA-ES (w tym wersji z heurystyką restartu IPOP) oraz porównanie ich działania z klasycznym CMA-ES. Testy zostały przeprowadzone na funkcjach benchmarkowych BBOB przy użyciu frameworka COCO.

W ramach projektu:

- zaimplementowano MA-ES z wariantami restartu,
- zachowano kompatybilność z oryginalnym interfejsem `cocoex`,
- wykonano eksperymenty porównawcze i analizę ERT oraz sukcesów osiągania targetu,
- wygenerowano wykresy runtime profile i tabele porównawcze.

## Uruchomienie

Aby uruchomić eksperymenty, wystarczy wykonać:

```bash
python experiment.py
```
