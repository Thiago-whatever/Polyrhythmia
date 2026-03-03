from pathlib import Path
import tensorflow as tf

def list_tags_in_event_file(event_path: Path):
    tags = set()
    n_events = 0
    try:
        for e in tf.compat.v1.train.summary_iterator(str(event_path)):
            n_events += 1
            if e.summary is None:
                continue
            for v in e.summary.value:
                if v.tag:
                    tags.add(v.tag)
    except Exception as ex:
        print(f"[ERROR] No pude leer {event_path.name}: {ex}")
        return None, 0

    return sorted(tags), n_events

def main():
    root = Path("logs/tensorboard")
    files = sorted(root.rglob("events.out.tfevents.*"))
    if not files:
        # fallback por si el nombre varía
        files = sorted(root.rglob("*tfevents*"))

    if not files:
        print("[ERROR] No encontré archivos tfevents bajo logs/tensorboard")
        return

    print(f"[OK] Encontré {len(files)} archivos tfevents\n")
    for f in files:
        tags, n = list_tags_in_event_file(f)
        print("—" * 80)
        print("FILE:", f)
        print("EVENTS:", n)
        if tags is None:
            continue
        print("TAGS:")
        for t in tags:
            print(" ", t)

if __name__ == "__main__":
    main()