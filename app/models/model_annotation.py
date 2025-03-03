from abc import ABC, abstractmethod
from datetime import datetime
import json

class ModelAnnotation(ABC):
    @abstractmethod
    def predict(self, text, app):
        pass

    def serialize(self, text, annotations, footer):
        """This function implements the Common Data Model v2"""
        output = {
            "nlp_output": {
                "record_metadata": {
                    "clinical_site_id": footer['provider_id'],
                    "patient_id": footer['person_id'],
                    "admission_id": footer['visit_detail_id'],
                    "record_id": footer['note_id'],
                    "record_type": footer['note_type_concept_id'],
                    "record_format": "json",
                    "record_creation_date": footer['note_datetime'],
                    "record_lastupdate_date": datetime.now().isoformat(),
                    "record_character_encoding": "UTF-8",
                    "record_extraction_date": datetime.now().isoformat(),
                    "report_section": footer['note_title'],
                    "report_language": "es",
                    "deidentified": "no",
                    "deidentification_pipeline_name": "",
                    "deidentification_pipeline_version": "",
                    "text": text,
                    "nlp_processing_date": datetime.now().isoformat(),
                    "nlp_processing_pipeline_name": self.__class__.__name__,
                    "nlp_processing_pipeline_version": "1.0",
                },
                "annotations": annotations
            },
            "nlp_service_info": {
                "service_app_name": "DT4H NLP Processor",
                "service_language": "en",
                "service_version": "1.0",
                "service_model": self.__class__.__name__
            }
        }
        output["nlp_output"]["processing_success"] = True
        return output
