# Dictionary Lookup NER API

This is a Flask API that performs Named Entity Recognition (NER) using a dictionary-based approach. The model relies on a CSV file containing entities and their corresponding labels.

## Prerequisites

Before starting, make sure you have Docker and Docker Compose installed on your system.

* Docker
* Docker Compose
## Instructions to Start the Service

1. Clone the repository
First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create the Entity CSV File
Make sure the spanish_entities.csv file exists in the root directory of the project. The file should follow this format:

```csv
valvulopatía,signs and symptoms,368009,SNOMED,2024-06-01
taquicardia,signs and symptoms,3424008,SNOMED,2024-06-02
presión arterial media,signs and symptoms,6797001,SNOMED,2024-06-03
ex-fumador,signs and symptoms,8517006,SNOMED,2024-06-04
lipoproteína de alta densidad,signs and symptoms,9422000,SNOMED,2024-06-05
This CSV file is used by the model to recognize entities in the text.
```

3. Build and Run the Docker Container
Build and run the service using Docker Compose:

```bash
docker compose up --build
```
This will start the service on port 8000.

4. Verify the Service is Running
You can open your browser or run a curl request to http://localhost:8000 to ensure the service is up and running.

## Available Endpoints

1. /process_text - Process a Single Text
Method: POST

This endpoint processes a single text and applies NER using the dictionary model.

Example Request

```bash
curl --location 'http://127.0.0.1:8000/process_bulk' \
--header 'Content-Type: application/json' \
--data '{ "content": [{
    "text": "ANTECEDENTES, ENFERMEDAD ACTUAL Y EXPLORACIÓN FÍSICA El caso que nos ocupa se centra en un varón 72 años que ingresa en la unidad coronaria (UCO) por tormenta arrítmica en noviembre de 2016. Los antecedentes personales de este paciente son: Independiente para actividades básicas de la vida diaria (ABVD). Fumador. Hipertensión arterial (HTA), dislipemia, diabetes mellitus tipo 2 en tratamiento con antidiabéticos orales (ADO). Enfermedad renal crónica (ERC) estadio 3A. Enfermedad pulmonar obstructiva crónica (EPOC) grave. Neoplasia vesical en 2011 tratado con cirugía (cistoprostatectomía + ureteroileostomía Bricker) y posteriormente QT. Historia cardiológica: cardiopatía isquémica crónica (infarto agudo de miocardio [IAM] en 1999) estable, con seguimiento por su médico de atención primaria. Como señalábamos previamente, el paciente ingresa en tormenta arrítmica. Para cese del problema arrítmico en el momento agudo, el paciente precisó de cinco cardioversiones y perfusión venosa continua de procainamida. Por persistencia de taquicardia ventricular no sostenida (TVNS) de similar morfología a la ya descrita, se sobreestimuló mediante electrocatéter femoral, solventándose la actividad arritmogénica. Durante su ingreso, se realiza coronariografía que demuestra cardiopatía isquémica difusa, con enfermedad de 3 vasos, sin lesiones agudas y desestimándose la cirugía cardiaca, dado el alto riesgo quirúrgico del paciente. Se deriva para estudio electrofisiológico (EEF) y ablación de taquicardia ventricular (TV) que, como complicación, presenta bloqueo auriculoventricular (AV) completo que persiste tras 1 semana del EEF, por lo que finalmente se decide implantar un desfibrilador automático implantable (DAI) bicameral (debido a que presenta cardiopatía isquémica sin lesiones agudas que pudieran explicar la tormenta arrítmica, con disfunción ventricular grave -FE4C Simpson 30%- y pronóstico vital superior a 12 meses) y es dado de alta con el siguiente tratamiento: adiro 100 mg (1 comprimido al día), bisoprolol 5 mg (1 comprimido por la mañana), atorvastatina 40 mg (1 comprimido por la noche), amiodarona 200 mg (cada 24 horas en la comida), indacaterol/glicopirronio inhalado (cada 24 horas), budesonida 200 mcg (cada 12 horas), hidroferol ampolla (bebible cada 10 días). El paciente permanece asintomático para angina y arritmias ventriculares durante 8 meses, hasta julio de 2017 cuando ingresa en medicina interna por síndrome febril a estudio."
}]}'
```
Example Response

```json
[
    {
        "nlp_output": {
            "annotations": [
                {
                    "concept_class": "signs and symptoms",
                    "concept_confidence": 0.95,
                    "concept_mention_string": "varón",
                    "controlled_vocabulary_concept_identifier": "",
                    "controlled_vocabulary_concept_official_term": "",
                    "controlled_vocabulary_namespace": "none",
                    "controlled_vocabulary_source": "original",
                    "controlled_vocabulary_version": "",
                    "dt4h_concept_identifier": "248153007",
                    "end_offset": 96,
                    "negation": "no",
                    "negation_confidence": 1.0,
                    "nel_component_type": "SNOMEDCT-ES",
                    "nel_component_version": "2024-03-31",
                    "ner_component_type": "dictionary lookup",
                    "ner_component_version": "3.7.0",
                    "qualifier_negation": "",
                    "qualifier_temporal": "",
                    "start_offset": 91
                },
                {
                    "concept_class": "signs and symptoms",
                    "concept_confidence": 0.95,
                    "concept_mention_string": "taquicardia",
                    "controlled_vocabulary_concept_identifier": "",
                    "controlled_vocabulary_concept_official_term": "",
                    "controlled_vocabulary_namespace": "none",
                    "controlled_vocabulary_source": "original",
                    "controlled_vocabulary_version": "",
                    "dt4h_concept_identifier": "3424008",
                    "end_offset": 1507,
                    "negation": "no",
                    "negation_confidence": 1.0,
                    "nel_component_type": "SNOMEDCT-ES",
                    "nel_component_version": "2024-03-31",
                    "ner_component_type": "dictionary lookup",
                    "ner_component_version": "3.7.0",
                    "qualifier_negation": "",
                    "qualifier_temporal": "",
                    "start_offset": 1496
                },
                {
                    "concept_class": "medication",
                    "concept_confidence": 0.95,
                    "concept_mention_string": "atorvastatina",
                    "controlled_vocabulary_concept_identifier": "",
                    "controlled_vocabulary_concept_official_term": "",
                    "controlled_vocabulary_namespace": "none",
                    "controlled_vocabulary_source": "original",
                    "controlled_vocabulary_version": "",
                    "dt4h_concept_identifier": "373444002",
                    "end_offset": 2079,
                    "negation": "no",
                    "negation_confidence": 1.0,
                    "nel_component_type": "SNOMEDCT-ES",
                    "nel_component_version": "2024-03-31",
                    "ner_component_type": "dictionary lookup",
                    "ner_component_version": "3.7.0",
                    "qualifier_negation": "",
                    "qualifier_temporal": "",
                    "start_offset": 2066
                }
            ],
            "processing_success": true,
            "record_metadata": {
                "admission_id": "",
                "clinical_site_id": "example_site",
                "deidentification_pipeline_name": "",
                "deidentification_pipeline_version": "",
                "deidentified": "no",
                "nlp_processing_date": "2024-09-30T12:48:39.438319",
                "nlp_processing_pipeline_name": "DictionaryLookupModel",
                "nlp_processing_pipeline_version": "1.0",
                "patient_id": "patient_id",
                "record_character_encoding": "UTF-8",
                "record_creation_date": "2024-09-30T12:48:39.438245",
                "record_extraction_date": "2024-09-30T12:48:39.438314",
                "record_format": "txt",
                "record_id": "patient_id",
                "record_lastupdate_date": "2024-09-30T12:48:39.438308",
                "record_type": "progress report",
                "report_language": "en",
                "report_section": "",
                "text": "ANTECEDENTES, ENFERMEDAD ACTUAL Y EXPLORACIÓN FÍSICA El caso que nos ocupa se centra en un varón 72 años que ingresa en la unidad coronaria (UCO) por tormenta arrítmica en noviembre de 2016. Los antecedentes personales de este paciente son: Independiente para actividades básicas de la vida diaria (ABVD). Fumador. Hipertensión arterial (HTA), dislipemia, diabetes mellitus tipo 2 en tratamiento con antidiabéticos orales (ADO). Enfermedad renal crónica (ERC) estadio 3A. Enfermedad pulmonar obstructiva crónica (EPOC) grave. Neoplasia vesical en 2011 tratado con cirugía (cistoprostatectomía + ureteroileostomía Bricker) y posteriormente QT. Historia cardiológica: cardiopatía isquémica crónica (infarto agudo de miocardio [IAM] en 1999) estable, con seguimiento por su médico de atención primaria. Como señalábamos previamente, el paciente ingresa en tormenta arrítmica. Para cese del problema arrítmico en el momento agudo, el paciente precisó de cinco cardioversiones y perfusión venosa continua de procainamida. Por persistencia de taquicardia ventricular no sostenida (TVNS) de similar morfología a la ya descrita, se sobreestimuló mediante electrocatéter femoral, solventándose la actividad arritmogénica. Durante su ingreso, se realiza coronariografía que demuestra cardiopatía isquémica difusa, con enfermedad de 3 vasos, sin lesiones agudas y desestimándose la cirugía cardiaca, dado el alto riesgo quirúrgico del paciente. Se deriva para estudio electrofisiológico (EEF) y ablación de taquicardia ventricular (TV) que, como complicación, presenta bloqueo auriculoventricular (AV) completo que persiste tras 1 semana del EEF, por lo que finalmente se decide implantar un desfibrilador automático implantable (DAI) bicameral (debido a que presenta cardiopatía isquémica sin lesiones agudas que pudieran explicar la tormenta arrítmica, con disfunción ventricular grave -FE4C Simpson 30%- y pronóstico vital superior a 12 meses) y es dado de alta con el siguiente tratamiento: adiro 100 mg (1 comprimido al día), bisoprolol 5 mg (1 comprimido por la mañana), atorvastatina 40 mg (1 comprimido por la noche), amiodarona 200 mg (cada 24 horas en la comida), indacaterol/glicopirronio inhalado (cada 24 horas), budesonida 200 mcg (cada 12 horas), hidroferol ampolla (bebible cada 10 días). El paciente permanece asintomático para angina y arritmias ventriculares durante 8 meses, hasta julio de 2017 cuando ingresa en medicina interna por síndrome febril a estudio."
            }
        },
        "nlp_service_info": {
            "service_app_name": "DT4H NLP Processor",
            "service_language": "en",
            "service_model": "DictionaryLookupModel",
            "service_version": "1.0"
        }
    }
]
```
2. /process_bulk - Process Multiple Texts
Method: POST

This endpoint processes multiple texts at once, applying NER using the dictionary model to each text.

## Environment Variables

You can configure the following environment variables in the Docker Compose file:

DICTIONARY_FILE: Path to the CSV file with dictionary entities (optional if using default ./english_entities.csv).
LANGUAGE: Language model to use (default: en).
Stopping the Service

## To stop the service, use:

```bash
docker-compose down
```
