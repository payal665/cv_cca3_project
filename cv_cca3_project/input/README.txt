PUT YOUR OWN ORIGINAL FILES IN THIS FOLDER (do not edit/crop/filter them):

  photo1.jpg   -> your original photograph (taken with your own phone/camera)
  photo2.jpg   -> a second photograph of the SAME object/scene
                  (take it from a slightly different angle/distance)
  video.mp4    -> your original 15-30 second video with a moving person/object

Allowed extensions: photos .jpg .jpeg .png ; video .mp4 .mov .avi .mkv .m4v
The file NAMES (photo1, photo2, video) must stay exactly like this.

Tips for good results:
  * photo1: a scene with a few objects, ideally some touching objects
    (coins, pens, fruits, cups) - this makes watershed segmentation meaningful.
    If photo1 has no touching objects, take one extra photo and pass it with
    --seg-image (see README).
  * Ideally one of your photos/video frames contains a face/person (for Haar + YOLO).
  * video: keep the camera still for part of the clip, someone/something moves.
