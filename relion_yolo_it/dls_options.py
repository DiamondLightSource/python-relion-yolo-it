"""
dls_options
-----------

Default options file for eBIC. Uses the DLS Hamilton GPU cluster.

Defaults for all options not defined here are as in the original relion_it.py script.
"""

#### crYOLO options

# Run Cryolo picking or autopicking
autopick_do_cryolo = True
# Threshold for cryolo autopicking (higher the threshold the more *discriminative the cryolo picker) ((* But beware it may still not be picking what you want! ))
cryolo_threshold = 0.3
# Finetune the cryolo general model by selecting good classes from 2D classification
cryolo_finetune = False

# Location of the cryolo specific files
cryolo_config = "/dls_sw/apps/EM/crYOLO/phosaurus_models/config.json"
cryolo_gmodel = "/dls_sw/apps/EM/crYOLO/phosaurus_models/gmodel_phosnet_202002_N63.h5"

# Running options for cryolo
cryolo_pick_gpus = "0 1 2 3"
cryolo_submit_to_queue = True
cryolo_queue_submission_template = "/dls_sw/apps/EM/crYOLO/qsub_cryolo_template_rh7"

do_second_pass = False

#### Other site configuration options

# Exectutable of UCSF MotionCor2
motioncor_exe = "/dls_sw/apps/EM/MotionCor2/1.3.1/MotionCor2"

# Executable to Kai Zhang's Gctf
gctf_exe = "/dls_sw/apps/EM/Gctf/1.18/Gctf"

# Executable for Alexis Rohou's CTFFIND4
ctffind4_exe = "/dls_sw/apps/EM/ctffind/4.1.13/bin/ctffind"

queue_submission_template = "/dls_sw/apps/EM/relion/qsub_template_hamilton"

motioncor_gpu = ""
motioncor_do_own = False
motioncor_mpi = 4
motioncor_threads = 10
motioncor_other_args = "--do_at_most 40"
motioncor_submit_to_queue = True

gctf_gpu = ""
ctffind_mpi = 4
use_ctffind_instead = False
ctffind_submit_to_queue = True

autopick_do_gpu = True
autopick_gpu = ""
autopick_mpi = 4
autopick_submit_to_queue = True

extract_mpi = 40
extract_submit_to_queue = True

batch_size = 20000

refine_do_gpu = True
refine_gpu = ""
refine_mpi = 5
refine_threads = 8
refine_scratch_disk = ""  # Hamilton nodes only have ~100 GB in /tmp, so don't use this
refine_submit_to_queue = True

minimum_batch_size = 250


#### Default options for the GUI

# (These might belong better in a different template file but they'll work here for now)

voltage = 300
Cs = 2.7
angpix = ""
import_images = "Movies/*.tif"
motioncor_doseperframe = ""
autopick_LoG_diam_max = ""
autopick_LoG_diam_min = ""
autopick_3dreference = ""
