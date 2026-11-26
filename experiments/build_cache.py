from experiments.lib.data import CACHE, build_cache

if __name__ == "__main__":
    n = build_cache(size=256, quality=92, workers=6)
    print(f"selesai: {n} gambar baru di {CACHE}")
