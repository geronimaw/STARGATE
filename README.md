# Progetto STARGATE
Repo per l'analisi dei movimenti in video di bambini con disturbi dello spettro autistico per la rilevazione di stereotipie motorie, in collaborazione con Il Faro Coperativa Sociale.

## Struttura del repository
Il repository contiene i seguenti file principali:
- `track.py`: Script Python per l'estrazione dei punti chiave dai video utilizzando OpenPose.
- `quant_movement.py`: Script Python per l'analisi quantitativa del movimento basato sui dati di tracciamento dei punti chiave estratti dai video.

## Requisiti
Per eseguire gli script, è necessario avere installato Python 3 e le librerie specificate in `requirements.txt`. Per installare le dipendenze, eseguire:
```pip install -r requirements.txt```

## Esempio
Il file `stereotipie_rilev.7z` contiene un esempio del risultato che vorremo ottenere dall'analisi di un video del dataset raccolto con il Faro. Il file è crittografato in formato 7z e può essere estratto utilizzando software come 7-Zip con la password fornita separatamente.