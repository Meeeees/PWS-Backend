# PWS Gezichtsherkenning Backend programma
## Dit is de backend repositorie van het PWS van Mees en Stijn over gezichtherkenning, bekijk de frontend repositorie [hier](https://github.com/Meeeees/PWS-Frontend-React)

### Starten van het programma
- allereerst moet je de "virtual enviroment" activeren, omdat de raspberry pi een linux gebaseerd systeem heeft doe je dit door het commando `source venv/bin/activate` uit te voeren.
- vervolgens kan je de fastapi server starten (in de development mode), dit doe je door het commando `fastapi dev main.py --host 0.0.0.0` uit te voeren


## Gebruikers flow
![Overzicht van de systeem architectuur](https://i.postimg.cc/FRTxPFKv/image.png)
De gebruikers flow van ons project is erg simpel in deze versie, momenteel kan er alleen nog video worden bekeken en een verzoek voor gezichtherkenning worden verstuurd.

## Technologieën
### hardware
- Raspberry Pi 4
- Pi camera
### Software
- Python 3.11
- deepface
- fastapi
- picamera2 
- numpy
