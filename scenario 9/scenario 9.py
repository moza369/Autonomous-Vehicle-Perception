import cv2
import numpy as np
import sounddevice as sd

# from moviepy import VideoFileClip

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# video = cv2.VideoCapture("ambulance.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

bleu_bas = np.array([100, 150, 50])
bleu_haut = np.array([140, 255, 255])
rouge_bas_1 = np.array([0, 150, 50])
rouge_haut_1 = np.array([10, 255, 255])
rouge_bas_2 = np.array([170, 150, 50])
rouge_haut_2 = np.array([180, 255, 255])

# clip = VideoFileClip("ambulance.mp4")
# son = clip.audio.to_soundarray(fps=22050).mean(axis=1)

sr = 22050
volume_micro = 0.0
audio_ok = True

def capter_micro(indata, frames, time, status):
    global volume_micro
    volume_micro = float(np.sqrt(np.mean(indata ** 2)))

try:
    flux_audio = sd.InputStream(channels=1, samplerate=sr, callback=capter_micro)
    flux_audio.start()
except Exception:
    audio_ok = False

fenetre = max(10, int(fps))
seuil_alternance = 2
base_volume = 0.01

historique = []
vehicule_signale = False
decalage = 0
num_frame = 0

while True:
    ok, frame = video.read()
    if not ok:
        break

    frame = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masque_bleu = cv2.inRange(hsv, bleu_bas, bleu_haut)
    masque_rouge = cv2.bitwise_or(
        cv2.inRange(hsv, rouge_bas_1, rouge_haut_1),
        cv2.inRange(hsv, rouge_bas_2, rouge_haut_2)
    )

    for masque, couleur in ((masque_rouge, (0, 0, 255)), (masque_bleu, (255, 0, 0))):
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 15 and h > 15:
                cv2.rectangle(frame, (x, y), (x + w, y + h), couleur, 2)

    pixels_rouge = cv2.countNonZero(masque_rouge)
    pixels_bleu = cv2.countNonZero(masque_bleu)

    if pixels_rouge < 50 and pixels_bleu < 50:
        dominante = 0
    elif pixels_rouge > pixels_bleu:
        dominante = 1
    else:
        dominante = -1

    historique.append(dominante)
    if len(historique) > fenetre:
        historique.pop(0)

    alternances = 0
    derniere = 0
    for d in historique:
        if d == 0:
            continue
        if derniere != 0 and d != derniere:
            alternances += 1
        derniere = d
    clignote = alternances >= seuil_alternance

    sirene = False
    if audio_ok:
        sirene = volume_micro > base_volume * 3 and volume_micro > 0.015
        if not sirene:
            base_volume = base_volume * 0.98 + volume_micro * 0.02

    
    # t = num_frame / fps
    # debut = int(t * sr)
    # fin = debut + int(0.5 * sr)
    # morceau = son[debut:fin]
    # if len(morceau) > 0:
    #     sirene = np.sqrt(np.mean(morceau ** 2)) > 0.042

    if clignote and sirene:
        vehicule_signale = True

    if vehicule_signale:
        decalage = min(150, decalage + 4)
        cv2.putText(frame, "ACTION: EMERGENCY VEHICLE DETECTED - PULLING OVER TO RIGHT SHOULDER",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.arrowedLine(frame, (280, 320), (280 + decalage, 320), (0, 0, 255), 3)

    cv2.imshow("Camera arriere", frame)
    num_frame += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

if audio_ok:
    flux_audio.stop()
video.release()
cv2.destroyAllWindows()
