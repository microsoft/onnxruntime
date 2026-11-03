from build_gdn_value8 import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="gdn-value8-v1", experiment_flag=FLAG, stamp="20261030",
         parent_variant="dense-block32-v2", parent_stamp="20261030", chunk=1024,
         program_name="GatedDeltaNetValue8", original_program_name="GatedDeltaNet", dispatches=36)
