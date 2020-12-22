def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """

    metadata = None

    if len(known_face_encodings) == 0:
        return metadata

    print(known_face_encodings[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:

        metadata = known_face_metadata[best_match_index]

    return metadata
