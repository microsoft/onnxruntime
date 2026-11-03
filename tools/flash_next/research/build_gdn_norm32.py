from build_gdn_tree32 import main


FLAG = "ORT_WEBGPU_GDN_NORM32"
PROGRAM = "GatedDeltaNetNorm32"


if __name__ == "__main__":
    main(norm_only=True)
