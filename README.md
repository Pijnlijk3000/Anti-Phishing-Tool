English

This project is an email scanner that scans the body (the text) of an email. Based on the content of the email, the scanner will determine if the email is spam or not.

HOW TO USE THE PROJECT
1. Download Python from the official website: https://www.python.org/downloads/
2. Download a code editor (e.g. Visual Studio Code, PyCharm, etc.)
3. Open the project in the code editor
NOW OPEN THE DATA FOLDER AND DOWNLOAD THE DATASET FROM KAGGLE. THE LINK IS IN THE README FILE IN THE DATA FOLDER. THE DATASET IS CALLED "phishing_email.csv" AND IS 106.6MB IN SIZE. TO USE THE DATASET, DOWNLOAD IT AND PLACE IT IN THE "data" DIRECTORY OF THIS REPOSITORY. SEE THE README FILE IN THE DATA FOLDER FOR MORE INFORMATION.
4. Open a terminal in the code editor
5. Install the libraries in the requirements.txt file by running the command: pip install -r requirements.txt OR py -m pip install -r requirements.txt depending on your Python installation 
6. Run the command: python phishingdetector.py
7. You will be prompted to enter the email you want to scan. Enter the email and press Enter.
8. The scanner will determine if the email is spam or not and display the result. It will also display the confidence level of the prediction.
9. When the confidence is LOW, the email is likely not spam. When the confidence is HIGH, the email is likely spam.
10. You can scan multiple emails by repeating steps 7-9. After each scan, you will be prompted to scan another email. You can simply paste the email and press Enter to scan another email.
11. To exit the scanner, type 'quit' IN LOWERCASE and press Enter. Or you either kill the terminal or press Ctrl + C.

DISCLAIMER: This project is in the early stages of development. The email scanner is not perfect and will not work well with short emails. The scanner is more accurate with longer emails. The scanner is also not perfect and may not always correctly determine if an email is spam or not. The scanner is a simple proof of concept and should not be used in a production environment YET.

If you want to train the model again, you can simply delete the pickle file located in the models folder. The model will be retrained when you run the modeltraining.py file. You can modify the amount of emails the model is trained on by changing the value of the sample_size variable in the modeltraining.py file. The default value is 10000 emails. Its located at line 31 in the modeltraining.py file.
```py
def load_and_preprocess_data(self, data_path, sample_size=10000):
```

Nederlands

Dit project is een e-mailscanner die de inhoud (de tekst) van een e-mail scant. Op basis van de inhoud van de e-mail bepaalt de scanner of de e-mail spam is of niet.

INSTALLATIE VAN HET PROJECT
1. Download Python van de officiële website: https://www.python.org/downloads/
2. Download een code-editor (bijv. Visual Studio Code, PyCharm, etc.)
3. Open het project in de code-editor
OPEN DE DATA MAP EN DOWNLOAD DE DATASET VAN KAGGLE. DE LINK IS IN DE README FILE IN DE DATA MAP. DE DATASET HEET "phishing_email.csv" EN IS 106.6MB GROOT. OM DE DATASET TE GEBRUIKEN, DOWNLOAD JE DEZE EN PLAATS JE DEZE IN DE "data" DIRECTORY VAN DEZE REPOSITORY. ZIE DE README FILE IN DE DATA MAP VOOR MEER INFORMATIE.
4. Open een terminal in de code-editor
5. Installeer de benodigde python libraries in het bestand requirements.txt door het commando uit te voeren: pip install -r requirements.txt OF py -m pip install -r requirements.txt, afhankelijk van de Python-installatie
6. Voer het commando uit: python phishingdetector.py
7. Het programma vraagt om de e-mail in te voeren. Voer de e-mail in en druk op Enter.
8. De scanner bepaalt of de e-mail spam is of niet en geeft het resultaat weer. Het zal ook het vertrouwensniveau van de voorspelling weergeven.
9. Wanneer het vertrouwen LAAG is, is de e-mail waarschijnlijk geen spam. Wanneer het vertrouwen HOOG is, is de e-mail waarschijnlijk spam.
10. Er kunnen meerdere e-mails gescand worden door de stappen 7 tot en met 9 te herhalen.
11. Om de scanner te verlaten, typ 'quit' in kleine letters en druk op Enter. Of je kunt de terminal afsluiten of op Ctrl + C drukken.

DISCLAIMER: Dit project is in de beginfase van ontwikkeling. De e-mailscanner is niet perfect en werkt niet goed met korte e-mails. De scanner is nauwkeuriger met langere e-mails. De scanner is ook niet perfect en kan niet altijd correct bepalen of een e-mail spam is of niet. De scanner is een eenvoudig proof of concept en mag nog niet in een productieomgeving worden gebruikt.
NOG EEN DISCLAIMER: De e-mailscanner is getraind op een dataset van e-mails die in het Engels zijn geschreven. De scanner werkt dus alleen met e-mails die in het Engels zijn geschreven.

Als je het model opnieuw wilt trainen, kun je eenvoudig het pickle-bestand verwijderen dat zich in de models-map bevindt. Het model wordt opnieuw getraind wanneer je het bestand modeltraining.py uitvoert. Je kunt de hoeveelheid e-mails waarop het model wordt getraind aanpassen door de waarde van de sample_size-variabele in het bestand modeltraining.py te wijzigen. De standaardwaarde is 10000 e-mails. Het bevindt zich op regel 31 in het bestand modeltraining.py.
```py
def load_and_preprocess_data(self, data_path, sample_size=10000):
```



