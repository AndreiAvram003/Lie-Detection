import os
import shutil
import re  # Pentru parsarea textului de mapare

# --- CONFIGURARE ---
# !!! ACTUALIZEAZĂ ACESTE CĂI ÎNAINTE DE A RUCA SCRIPTUL !!!
SOURCE_BASE_DIR = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\face_analysis\cropped_faces"  # Directorul care conține Atul02459, Dishant10480 etc.
DEST_BASE_DIR = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\face_analysis\new_faces"  # Unde vor fi create noile directoare Atul1, Dishant1 etc.
# Numele frame-ului din fiecare director sursă individual
SOURCE_FRAME_NAME_IN_SUBDIR = "frame_00000.jpg"
# --- SFÂRȘIT CONFIGURARE ---

# Textul cu regulile de mapare furnizat de tine
# Am corectat "Mansi0080" la "Mansi00080" pentru a se potrivi cu formatarea de 5 cifre.
# Am corectat "Sailja2030" la "Sailja02030" pentru a se potrivi cu formatarea de 5 cifre.
# Am corectat "Harsha11900" la "Harsha11900" (era deja corect, doar am verificat ca prefixul să fie consistent)
raw_mappings_text = """
de la Atul13130 pana la Atul13420 sa fie Atul1
de la Atul12600 pana la Atul12860 sa fie Atul2
de la Atul11870 pana la Atul12060 sa fie Atul3
de la Atul12240 pana la Atul12330 sa fie Atul4
de la Dishant10480 pana la Dishant10620 sa fie Dishant1
de la Dishant10150 pana la Dishant10300 sa fie Dishant2
de la Dishant09545 pana la Dishant09710 sa fie Dishant3
de la Dishant09900 pana la Dishant10140 sa fie Dishant4
de la Dishant10850 pana la Dishant11330 sa fie Dishant5
de la DrPrashant03040 pana la DrPrashant03130 sa fie DrPrashant1
de la DrPrashant02580 pana la DrPrashant02970 sa fie DrPrashant2
de la DrPrashant01870 pana la DrPrashant02000 sa fie DrPrashant3
de la DrPrashant02260 pana la DrPrashant02430 sa fie DrPrashant4
de la DrPrashant03390 pana la DrPrashant03990 sa fie DrPrashant5
de la Harsha08660 pana la Harsha08800 sa fie Harsha1
de la Harsha08260 pana la Harsha08450 sa fie Harsha2
de la Harsha07410 pana la Harsha07800 sa fie Harsha3
de la Harsha07910 pana la Harsha08140 sa fie Harsha4
de la Harsha09070 pana la Harsha09580 sa fie Harsha5
de la Manasvi01450 pana la Manasvi01650 sa fie Manasvi1
de la Manasvi01800 pana la Manasvi02035 sa fie Manasvi2
de la Manasvi02280 pana la Manasvi02640 sa fie Manasvi3
de la Manasvi04380 pana la Manasvi04680 sa fie Manasvi4
de la Mansi00080 pana la Mansi01140 sa fie Mansi1
de la Mansi01560 pana la Mansi01760 sa fie Mansi2
de la Mansi01800 pana la Mansi02070 sa fie Mansi3
de la Mansi02460 pana la Mansi02940 sa fie Mansi4
de la Mansi03000 pana la Mansi03470 sa fie Mansi5
de la Sailja01320 pana la Sailja01470 sa fie Sailja1
de la Sailja01570 pana la Sailja01700 sa fie Sailja2
de la Sailja01900 pana la Sailja02030 sa fie Sailja3
de la Sailja02100 pana la Sailja02270 sa fie Sailja4
de la Sailja03960 pana la Sailja04290 sa fie Sailja5
de la Atul02440 pana la Atul03085 sa fie Atul5
de la Atul04380 pana la Atul04550 sa fie Atul6
de la Atul04760 pana la Atul05450 sa fie Atul7
de la Atul05790 pana la Atul06030 sa fie Atul8
de la Dishant03370 pana la Dishant03990 sa fie Dishant6
de la Dishant04520 pana la Dishant04680 sa fie Dishant7
de la Dishant05750 pana la Dishant05940 sa fie Dishant8
de la Dishant06100 pana la Dishant06220 sa fie Dishant9
de la Dishant06500 pana la Dishant06900 sa fie Dishant10
de la DrPrashant04940 pana la DrPrashant05200 sa fie DrPrashant6
de la DrPrashant05340 pana la DrPrashant05500 sa fie DrPrashant7
de la DrPrashant05600 pana la DrPrashant05850 sa fie DrPrashant8
de la DrPrashant05970 pana la DrPrashant06150 sa fie DrPrashant9
de la DrPrashant06300 pana la DrPrashant06690 sa fie DrPrashant10
de la Harsha11400 pana la Harsha11900 sa fie Harsha6
de la Harsha12090 pana la Harsha12210 sa fie Harsha7
de la Harsha12350 pana la Harsha12700 sa fie Harsha8
de la Harsha12830 pana la Harsha12950 sa fie Harsha9
de la Harsha13210 pana la Harsha13670 sa fie Harsha10
de la Manasvi06420 pana la Manasvi06900 sa fie Manasvi5
de la Manasvi07080 pana la Manasvi07410 sa fie Manasvi6
de la Manasvi07540 pana la Manasvi07660 sa fie Manasvi7
de la Manasvi07780 pana la Manasvi07950 sa fie Manasvi8
de la Mansi04200 pana la Mansi04390 sa fie Mansi6
de la Mansi04650 pana la Mansi04820 sa fie Mansi7
de la Mansi04920 pana la Mansi05080 sa fie Mansi8
de la Mansi05160 pana la Mansi05390 sa fie Mansi9
de la Mansi06000 pana la Mansi06270 sa fie Mansi10
de la Sailja05170 pana la Sailja05520 sa fie Sailja6
de la Sailja06360 pana la Sailja06500 sa fie Sailja7
de la Sailja06620 pana la Sailja06730 sa fie Sailja8
de la Sailja06890 pana la Sailja07040 sa fie Sailja9
de la Sailja07300 pana la Sailja07680 sa fie Sailja10
"""


def parse_source_id_string(source_id_str):
    """Parsează un string de tipul 'PrefixNumar' (ex: 'Atul13130') în ('Prefix', Numar_int)."""
    match = re.match(r"([a-zA-Z_]+)(\d+)", source_id_str)
    if match:
        prefix_part = match.group(1)
        number_part_str = match.group(2)
        return prefix_part, int(number_part_str)
    raise ValueError(f"Nu s-a putut parsa ID-ul sursă: {source_id_str}. Așteptat format 'PrefixNumar'.")


parsed_mappings = []
for line_num, line in enumerate(raw_mappings_text.strip().split('\n')):
    line = line.strip()
    if not line:
        continue

    # Regex pentru a captura: "de la (ID_Start_Complet) pana la (ID_Stop_Complet_SAU_Numar_Stop) sa fie (Nume_Output)"
    pattern = r"de la\s+([a-zA-Z_0-9]+)\s+pana la\s+([a-zA-Z_0-9]+)\s+sa fie\s+([a-zA-Z_0-9]+)"
    match = re.match(pattern, line)

    if not match:
        print(f"Avertisment (linia {line_num + 1}): Nu s-a putut parsa linia: '{line}'")
        continue

    full_start_id_str, end_id_or_num_str, output_name = match.groups()

    try:
        start_prefix, start_num_int = parse_source_id_string(full_start_id_str)

        # Verifică dacă al doilea ID conține prefix sau este doar număr
        try:
            end_prefix_check, end_num_int = parse_source_id_string(end_id_or_num_str)
            # Dacă parsarea reușește și prefixul este diferit, ar putea fi o problemă (dar de obicei e același)
            if start_prefix != end_prefix_check:
                print(
                    f"Avertisment (linia {line_num + 1}): Prefixe diferite detectate ('{start_prefix}' vs '{end_prefix_check}') pentru '{line}'. Se va folosi prefixul de start: '{start_prefix}'.")
        except ValueError:  # Probabil este doar un număr pentru partea de final
            end_num_int = int(end_id_or_num_str)  # Presupune că prefixul este același cu cel de start

        parsed_mappings.append({
            "output_name": output_name,
            "prefix": start_prefix,
            "start_num": start_num_int,
            "end_num": end_num_int
        })
    except ValueError as e:
        print(f"Eroare la procesarea liniei {line_num + 1} ('{line}'): {e}")

# Asigură-te că directorul de destinație există
os.makedirs(DEST_BASE_DIR, exist_ok=True)
print(f"Se vor crea directoarele organizate în: {DEST_BASE_DIR}\n")

total_clips_processed = 0
total_frames_copied = 0

for mapping_rule in parsed_mappings:
    output_clip_name = mapping_rule["output_name"]
    source_prefix = mapping_rule["prefix"]
    start_number = mapping_rule["start_num"]
    end_number = mapping_rule["end_num"]

    current_output_clip_dir = os.path.join(DEST_BASE_DIR, output_clip_name)
    os.makedirs(current_output_clip_dir, exist_ok=True)

    print(
        f"Procesare pentru clipul de ieșire: '{output_clip_name}' (Sursă: {source_prefix}{start_number:05d} - {source_prefix}{end_number:05d})")

    output_frame_index = 0
    frames_in_this_clip = 0

    # Iterează prin intervalul numeric specificat
    for current_num_idx in range(start_number, end_number + 1):
        # Construiește numele directorului sursă, presupunând zero-padding la 5 cifre pentru partea numerică
        # Ex: dacă source_prefix="Atul" și current_num_idx=2459, source_dir_name="Atul02459"
        # Ex: dacă source_prefix="Atul" și current_num_idx=13130, source_dir_name="Atul13130"
        source_dir_name_numeric_part = f"{current_num_idx:05d}"
        source_dir_name = f"{source_prefix}{source_dir_name_numeric_part}"

        source_frame_full_path = os.path.join(SOURCE_BASE_DIR, source_dir_name, SOURCE_FRAME_NAME_IN_SUBDIR)

        if os.path.exists(source_frame_full_path):
            # Construiește numele frame-ului de destinație cu zero-padding
            dest_frame_filename = f"{output_frame_index:05d}.jpg"
            dest_frame_full_path = os.path.join(current_output_clip_dir, dest_frame_filename)

            try:
                shutil.copy2(source_frame_full_path, dest_frame_full_path)
                output_frame_index += 1
                frames_in_this_clip += 1
            except Exception as e:
                print(f"  -> Eroare la copierea '{source_frame_full_path}' la '{dest_frame_full_path}': {e}")
        else:
            # Acest print poate fi foarte stufos dacă sunt multe numere lipsă în interval.
            # Poți alege să-l comentezi dacă nu ai nevoie de detalii pentru fiecare fișier lipsă.
            # print(f"  -> Avertisment: Fișierul sursă nu există: {source_frame_full_path}")
            pass

    if frames_in_this_clip > 0:
        print(
            f"  => Finalizat pentru '{output_clip_name}'. {frames_in_this_clip} frame-uri copiate în '{current_output_clip_dir}'.")
        total_clips_processed += 1
        total_frames_copied += frames_in_this_clip
    else:
        print(
            f"  => AVERTISMENT: Nu s-au găsit sau copiat frame-uri pentru '{output_clip_name}' din intervalul specificat.")
        # Opțional, poți șterge directorul gol creat
        try:
            if not os.listdir(current_output_clip_dir):  # Verifică dacă e gol
                os.rmdir(current_output_clip_dir)
                # print(f"    Directorul gol '{current_output_clip_dir}' a fost șters.")
        except OSError as e:
            print(f"    Eroare la ștergerea directorului gol '{current_output_clip_dir}': {e}")

print(f"\n--- Procesul de reorganizare a folderelor a fost finalizat ---")
print(f"Total clipuri procesate și create: {total_clips_processed}")
print(f"Total frame-uri copiate: {total_frames_copied}")
print(f"Verifică directoarele create în: {DEST_BASE_DIR}")