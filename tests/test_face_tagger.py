from src.face_tagger.face_tagger import face_tagger


def test_face_tagger():
    assert face_tagger() == "Hello Face Tagger"
