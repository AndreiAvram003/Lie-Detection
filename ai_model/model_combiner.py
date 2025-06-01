import numpy as np

def combine_predictions_fuzzy(face_pred, face_conf, voice_pred, voice_conf):
    """
    Combină predicțiile față + voce folosind o logică fuzzy simplificată.
    Returnează clasa finală și nivelul de încredere.
    """

    # Dacă una dintre predicții e incertă, scade încrederea generală
    if face_pred == -1 or voice_pred == -1:
        return -1, 0.0

    # Reguli fuzzy simple:
    if face_pred == voice_pred:
        # Când ambele metode spun același lucru => combinăm încrederea
        combined_conf = (face_conf + voice_conf) / 2 * 100
        return face_pred, combined_conf
    else:
        # Când metodele sunt în conflict, alegem predicția cu încrederea mai mare
        if face_conf > voice_conf:
            difference_conf = 1 - voice_conf
            confidence = (face_conf + difference_conf) / 2
            return face_pred, confidence
        else:
            difference_conf = 1 - face_conf
            confidence = (voice_conf + difference_conf) / 2
            return voice_pred
