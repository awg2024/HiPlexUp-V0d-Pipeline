#!/usr/bin/env Rscript

# ============================================================
# HiPlexUp V0d downstream analysis
# ============================================================
#
# Purpose
# -------
# Continue the original HiPlexUp workflow using the NEW Python outputs:
#
#   row-wise-139/*.csv
#   row-wise-141/*.csv
#       -> per-cell outputs from measure_cells.py
#
#   tissue_masks/Slide_139/*.json
#   tissue_masks/Slide_141/*.json
#       -> tissue ROI JSONs from tissue_roi.py
#          (mask_pixels, shape_yx, vertices_xy, source_tiff)
#
# Python has already performed:
#   - segmentation
#   - area QC
#   - manual V0d / non-V0d classification
#   - background correction
#   - per-cell fluorescence measurement
#
# Therefore R does NOT re-threshold or re-classify cells.
#
# R performs:
#   1. Load both slides
#   2. Match each sample to its tissue ROI JSON
#   3. Add animal/group metadata (optional file; safe fallback included)
#   4. Normalize spatial coordinates using the tissue ROI
#   5. Quantify raw counts, tissue-normalized counts, V0d fraction
#   6. Calculate one fluorescence median per mouse
#   7. Run independent two-group Mann-Whitney/Wilcoxon rank-sum tests
#   8. Apply Holm correction across the five fluorescence markers
#   9. Produce spatial and mouse-level summary plots
#
# IMPORTANT
# ---------
# Current experimental assumption:
#   ONE SAMPLE / SCAN = ONE DIFFERENT MOUSE.
#
# If that changes, the animal aggregation must be changed.
#
# Tissue density:
#   The ROI JSON gives mask area in PIXELS. Therefore this script always
#   calculates cells per million tissue-mask pixels.
#
#   If physical pixel size is known, set PIXEL_SIZE_X_UM and
#   PIXEL_SIZE_Y_UM below and true cells/mm^2 will also be calculated.
#
# Spatial normalization:
#   The tissue-mask polygon is sufficient for ROI size/position
#   normalization. It is NOT a corpus-callosum landmark and does not
#   reproduce the old manual CC rotation. That can be added later for
#   final anatomical spatial registration without changing the count or
#   fluorescence analysis.
# ============================================================


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------


DATA_ROOT <- "/Users/angusgray/Desktop/V0d-Histology"

PROJECT_ROOT <- file.path(
  DATA_ROOT,
  "HiPlexUp-V0d-Pipeline"
)


SLIDES <- c("139", "141")

ROW_WISE_DIRS <- c(
  "139" = file.path("/Users/angusgray/Desktop/V0d-Histology/row-wise-139"),
  "141" = file.path("/Users/angusgray/Desktop/V0d-Histology/row-wise-141")
)

# TISSUE_MASK_DIRS <- c(
#   "139" = file.path("/Users/angusgray/Desktop/V0d-Histology/tissue_masks_139"),
#   "141" = file.path("/Users/angusgray/Desktop/V0d-Histology/tissue_masks_141")
# )

OUTPUT_DIR <- file.path(DATA_ROOT, "r_analysis")

# Optional metadata file.
# Expected columns:
#   Sample, Animal, Group
# Optional fourth column:
#   Slide
#
# If this file does not exist, the script still runs:
#   Animal = Sample
#   Group  = Slide_139 / Slide_141
METADATA_FILE <- file.path(DATA_ROOT, "animal_metadata.csv")



# Physical pixel size is not stored in the tissue ROI JSON supplied so far.
# Leave NA to use cells per million mask pixels.
# If known, enter micrometres per pixel here.
PIXEL_SIZE_X_UM <- 0.2270
PIXEL_SIZE_Y_UM <- 0.2270


# ------------------------------------------------------------
# PACKAGES
# ------------------------------------------------------------

required_packages <- c(
  "dplyr",
  "readr",
  "tidyr",
  "ggplot2",
  "jsonlite"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing R package(s): ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them with:\ninstall.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))"
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
})


# ------------------------------------------------------------
# OUTPUT FOLDERS
# ------------------------------------------------------------

TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
PNG_DIR   <- file.path(OUTPUT_DIR, "figures", "png")
SVG_DIR   <- file.path(OUTPUT_DIR, "figures", "svg")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PNG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(SVG_DIR, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

CLASSIFICATIONS <- c("V0d-cell", "non-V0d-cell")

CHANNELS <- c(
  "DAPI",
  "EVX1",
  "PAX2",
  "DBX1",
  "VGAT"
)

FLUORESCENCE_COLUMNS <- paste0(
  CHANNELS,
  "_mean_corrected"
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

canonical_sample_key <- function(x) {
  x <- basename(as.character(x))

  # File extension
  x <- sub("\\.(tif|tiff|csv|json)$", "", x, ignore.case = TRUE)

  # Known pipeline suffixes.
  # Repeated stripping makes matching tolerant of filenames generated
  # at different pipeline stages.
  suffixes <- c(
    "_tissue_roi",
    "_tissue_mask",
    "_cellpose_mask",
    "_cellpose_overlay",
    "_cellpose_run",
    "_cellpose",
    "_classified_cells",
    "_classified",
    "_measurements",
    "_measurement",
    "_measured_cells",
    "_cells",
    "_background",
    "_ALLCHANNELS_16bit_FIJI",
    "_ALLCHANNELS_16bit"
  )

  changed <- TRUE
  while (changed) {
    old <- x
    for (suffix in suffixes) {
      x <- sub(
        paste0(suffix, "$"),
        "",
        x,
        ignore.case = TRUE
      )
    }
    changed <- !identical(old, x)
  }

  x
}


first_existing_column <- function(df, candidates) {
  hit <- candidates[candidates %in% names(df)]

  if (length(hit) == 0) {
    return(NA_character_)
  }

  hit[[1]]
}


save_plot <- function(plot_object, name, width = 6, height = 5) {
  ggsave(
    filename = file.path(PNG_DIR, paste0(name, ".png")),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    bg = "white"
  )

  ggsave(
    filename = file.path(SVG_DIR, paste0(name, ".svg")),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    bg = "white"
  )
}


safe_median <- function(x) {
  x <- x[is.finite(x)]

  if (length(x) == 0) {
    return(NA_real_)
  }

  median(x)
}


# ------------------------------------------------------------
# LOAD measure_cells.py OUTPUTS
# ------------------------------------------------------------

load_slide_csvs <- function(slide, folder) {

  if (!dir.exists(folder)) {
    stop("Missing input directory: ", folder)
  }

  csv_files <- list.files(
    folder,
    pattern = "\\.csv$",
    full.names = TRUE,
    recursive = FALSE
  )

  # Avoid accidentally reading outputs/summaries if they are ever placed
  # inside row-wise-*.
  csv_files <- csv_files[
    !grepl(
      "statistics|counts|summary",
      basename(csv_files),
      ignore.case = TRUE
    )
  ]

  if (length(csv_files) == 0) {
    stop("No input CSV files found in: ", folder)
  }

  message(
    "Slide ", slide, ": loading ",
    length(csv_files), " measure_cells CSV file(s)"
  )

  pieces <- lapply(
    csv_files,
    function(path) {

      df <- read_csv(
        path,
        show_col_types = FALSE,
        progress = FALSE
      )

      df$Source_file <- basename(path)
      df$Slide <- as.character(slide)

      sample_col <- first_existing_column(
        df,
        c(
          "Image_ID",
          "Sample",
          "sample",
          "Image",
          "image_id"
        )
      )

      if (is.na(sample_col)) {
        # Safe fallback when each CSV corresponds to one sample.
        df$Sample <- tools::file_path_sans_ext(
          basename(path)
        )
      } else {
        df$Sample <- as.character(
          df[[sample_col]]
        )
      }

      df$Sample_key <- canonical_sample_key(
        df$Sample
      )

      df
    }
  )

  bind_rows(pieces)
}


slide_data <- lapply(
  SLIDES,
  function(slide) {
    load_slide_csvs(
      slide,
      ROW_WISE_DIRS[[slide]]
    )
  }
)

NT <- bind_rows(slide_data)

if (!"Classification" %in% names(NT)) {
  stop(
    "The measure_cells CSVs do not contain a 'Classification' column."
  )
}

NT$Classification <- trimws(
  as.character(NT$Classification)
)

NT <- NT %>%
  filter(
    Classification %in% CLASSIFICATIONS
  )

if (nrow(NT) == 0) {
  stop(
    "No V0d-cell or non-V0d-cell rows were found."
  )
}

missing_fluorescence <- setdiff(
  FLUORESCENCE_COLUMNS,
  names(NT)
)

if (length(missing_fluorescence) > 0) {
  stop(
    paste0(
      "Missing corrected fluorescence column(s): ",
      paste(missing_fluorescence, collapse = ", "),
      "\nThese should be outputs from measure_cells.py."
    )
  )
}


# # ------------------------------------------------------------
# # LOAD TISSUE ROI JSON FILES
# # ------------------------------------------------------------

# read_tissue_roi_json <- function(path, slide) {

#   obj <- tryCatch(
#     fromJSON(
#       path,
#       simplifyVector = TRUE
#     ),
#     error = function(e) NULL
#   )

#   if (is.null(obj)) {
#     return(NULL)
#   }

#   # Tissue ROI JSON must contain a top-level mask_pixels field.
#   if (is.null(obj$mask_pixels)) {
#     return(NULL)
#   }

#   source_name <- if (!is.null(obj$source_tiff)) {
#     obj$source_tiff
#   } else {
#     basename(path)
#   }

#   shape_y <- NA_real_
#   shape_x <- NA_real_

#   if (!is.null(obj$shape_yx) &&
#       length(obj$shape_yx) >= 2) {
#     shape_y <- as.numeric(obj$shape_yx[[1]])
#     shape_x <- as.numeric(obj$shape_yx[[2]])
#   }

#   x_min <- NA_real_
#   x_max <- NA_real_
#   y_min <- NA_real_
#   y_max <- NA_real_

#   if (!is.null(obj$vertices_xy)) {

#     vertices <- as.matrix(
#       obj$vertices_xy
#     )

#     if (ncol(vertices) >= 2 &&
#         nrow(vertices) >= 3) {

#       vx <- as.numeric(vertices[, 1])
#       vy <- as.numeric(vertices[, 2])

#       x_min <- min(vx, na.rm = TRUE)
#       x_max <- max(vx, na.rm = TRUE)
#       y_min <- min(vy, na.rm = TRUE)
#       y_max <- max(vy, na.rm = TRUE)
#     }
#   }

#   tibble(
#     Slide = as.character(slide),
#     ROI_json = basename(path),
#     ROI_source_tiff = basename(source_name),
#     Sample_key = canonical_sample_key(source_name),
#     mask_pixels = as.numeric(obj$mask_pixels),
#     shape_y = shape_y,
#     shape_x = shape_x,
#     x_min = x_min,
#     x_max = x_max,
#     y_min = y_min,
#     y_max = y_max,
#     roi_center_x = (x_min + x_max) / 2,
#     roi_center_y = (y_min + y_max) / 2,
#     roi_width_px = x_max - x_min,
#     roi_height_px = y_max - y_min
#   )
# }


# load_tissue_rois <- function(slide, folder) {

#   if (!dir.exists(folder)) {
#     stop("Missing tissue-mask directory: ", folder)
#   }

#   json_files <- list.files(
#     folder,
#     pattern = "\\.json$",
#     full.names = TRUE,
#     recursive = TRUE
#   )

#   if (length(json_files) == 0) {
#     stop(
#       "No JSON files found in tissue-mask directory: ",
#       folder
#     )
#   }

#   rows <- lapply(
#     json_files,
#     read_tissue_roi_json,
#     slide = slide
#   )

#   rows <- rows[
#     !vapply(rows, is.null, logical(1))
#   ]

#   if (length(rows) == 0) {
#     stop(
#       "JSON files were found in ",
#       folder,
#       " but none contained top-level mask_pixels."
#     )
#   }

#   result <- bind_rows(rows)

#   duplicate_keys <- result %>%
#     count(
#       Slide,
#       Sample_key,
#       name = "n"
#     ) %>%
#     filter(n > 1)

#   if (nrow(duplicate_keys) > 0) {
#     warning(
#       "Multiple tissue ROI JSONs matched the same sample key. ",
#       "Keeping the first JSON for each sample."
#     )

#     result <- result %>%
#       group_by(
#         Slide,
#         Sample_key
#       ) %>%
#       slice(1) %>%
#       ungroup()
#   }

#   result
# }


# tissue_list <- lapply(
#   SLIDES,
#   function(slide) {
#     load_tissue_rois(
#       slide,
#       TISSUE_MASK_DIRS[[slide]]
#     )
#   }
# )

# Tissue <- bind_rows(tissue_list)

# write_csv(
#   Tissue,
#   file.path(
#     TABLE_DIR,
#     "tissue_mask_summary.csv"
#   )
# )


# # ------------------------------------------------------------
# # MATCH CELL CSVs TO TISSUE ROI JSONs
# # ------------------------------------------------------------

# NT <- NT %>%
#   left_join(
#     Tissue,
#     by = c(
#       "Slide",
#       "Sample_key"
#     )
#   )

# unmatched <- NT %>%
#   filter(
#     is.na(mask_pixels)
#   ) %>%
#   distinct(
#     Slide,
#     Sample,
#     Sample_key,
#     Source_file
#   )

# if (nrow(unmatched) > 0) {

#   message(
#     "\nSamples that did not match a tissue ROI JSON:"
#   )

#   print(
#     unmatched,
#     n = nrow(unmatched)
#   )

#   stop(
#     paste0(
#       "Could not match all row-wise samples to tissue-mask JSONs. ",
#       "The script matches filenames after stripping pipeline suffixes. ",
#       "Check the names printed above."
#     )
#   )
# }

# message(
#   "Tissue ROI matching: PASS (",
#   n_distinct(NT$Sample_key),
#   " samples)"
# )


# ------------------------------------------------------------
# ANIMAL / GROUP METADATA
# ------------------------------------------------------------

if (file.exists(METADATA_FILE)) {

  message(
    "Loading metadata: ",
    METADATA_FILE
  )

  Metadata <- read_csv(
    METADATA_FILE,
    show_col_types = FALSE
  )

  metadata_sample_col <- first_existing_column(
    Metadata,
    c(
      "Sample",
      "Image_ID",
      "sample",
      "Image",
      "image_id"
    )
  )

  if (is.na(metadata_sample_col)) {
    stop(
      "animal_metadata.csv must contain a Sample or Image_ID column."
    )
  }

  if (!all(c("Animal", "Group") %in% names(Metadata))) {
    stop(
      "animal_metadata.csv must contain Animal and Group columns."
    )
  }

  Metadata$Sample_key <- canonical_sample_key(
    Metadata[[metadata_sample_col]]
  )

  Metadata$Animal <- as.character(
    Metadata$Animal
  )

  Metadata$Group <- as.character(
    Metadata$Group
  )

  if ("Slide" %in% names(Metadata)) {

    Metadata$Slide <- as.character(
      Metadata$Slide
    )

    Metadata <- Metadata %>%
      select(
        Slide,
        Sample_key,
        Animal,
        Group
      ) %>%
      distinct()

    NT <- NT %>%
      left_join(
        Metadata,
        by = c(
          "Slide",
          "Sample_key"
        )
      )

  } else {

    Metadata <- Metadata %>%
      select(
        Sample_key,
        Animal,
        Group
      ) %>%
      distinct()

    NT <- NT %>%
      left_join(
        Metadata,
        by = "Sample_key"
      )
  }

  metadata_missing <- NT %>%
    filter(
      is.na(Animal) |
      is.na(Group)
    ) %>%
    distinct(
      Slide,
      Sample,
      Sample_key
    )

  if (nrow(metadata_missing) > 0) {
    print(metadata_missing)
    stop(
      "Some samples did not match animal_metadata.csv."
    )
  }

} else {

  message(
    "\nNo animal_metadata.csv found."
  )

  message(
    "Using each Sample as Animal and Slide as Group."
  )

  # This is valid for the current design because every scan/sample is
  # from a different mouse.
  NT <- NT %>%
    mutate(
      Animal = Sample_key,
      Group = paste0(
        "Slide_",
        Slide
      )
    )

  metadata_template <- NT %>%
    distinct(
      Slide,
      Sample = Sample,
      Sample_key
    ) %>%
    mutate(
      Animal = Sample_key,
      Group = paste0(
        "Slide_",
        Slide
      )
    ) %>%
    select(
      Slide,
      Sample,
      Animal,
      Group
    )

  write_csv(
    metadata_template,
    file.path(
      TABLE_DIR,
      "animal_metadata_template.csv"
    )
  )
}


# ------------------------------------------------------------
# VERIFY CURRENT DESIGN: ONE SAMPLE = ONE MOUSE
# ------------------------------------------------------------

animal_scan_check <- NT %>%
  distinct(
    Animal,
    Sample_key
  ) %>%
  count(
    Animal,
    name = "n_samples"
  )

if (any(animal_scan_check$n_samples > 1)) {

  warning(
    paste0(
      "At least one Animal maps to multiple samples/scans. ",
      "The current script is designed for one scan per mouse. ",
      "Counts should be aggregated differently before final inference ",
      "if these are repeated sections from the same animal."
    )
  )
}
# ------------------------------------------------------------
# SPATIAL COORDINATES + PHYSICAL UNITS
# ------------------------------------------------------------

required_spatial_columns <- c(
  "X_centroid_px",
  "Y_centroid_px",
  "Area_px"
)

missing_spatial <- setdiff(
  required_spatial_columns,
  names(NT)
)

if (length(missing_spatial) > 0) {
  stop(
    paste(
      "Missing spatial columns:",
      paste(missing_spatial, collapse = ", ")
    )
  )
}

message(
  "Spatial columns detected: X_centroid_px, Y_centroid_px, Area_px"
)

NT <- NT %>%
  mutate(

    # Original measured coordinates
    X_raw_px = as.numeric(X_centroid_px),
    Y_raw_px = as.numeric(Y_centroid_px),
    Cell_area_px = as.numeric(Area_px),

    # Physical position
    X_um = X_raw_px * PIXEL_SIZE_X_UM,
    Y_um = Y_raw_px * PIXEL_SIZE_Y_UM,

    # Physical cell area
    Cell_area_um2 =
      Cell_area_px *
      PIXEL_SIZE_X_UM *
      PIXEL_SIZE_Y_UM
  )

SPATIAL_AVAILABLE <- TRUE

# ------------------------------------------------------------
# SAMPLE-LEVEL INFORMATION
# ------------------------------------------------------------

Sample_info <- NT %>%
  distinct(
    Slide,
    Group,
    Animal,
    Sample,
    Sample_key
  )

write_csv(
  Sample_info,
  file.path(
    TABLE_DIR,
    "sample_information.csv"
  )
)

# ------------------------------------------------------------
# IDENTIFY POPULATIONS
# ------------------------------------------------------------

V0d <- NT %>%
  filter(
    Classification == "V0d-cell"
  )

nonV0d <- NT %>%
  filter(
    Classification == "non-V0d-cell"
  )


# ------------------------------------------------------------
# QUANTIFICATION: COUNTS PER MOUSE
# ------------------------------------------------------------

count_long <- NT %>%
  count(
    Slide,
    Group,
    Animal,
    Sample_key,
    Classification,
    name = "Count"
  )

count_wide <- count_long %>%
  pivot_wider(
    names_from = Classification,
    values_from = Count,
    values_fill = 0
  )

if (!"V0d-cell" %in% names(count_wide)) {
  count_wide$`V0d-cell` <- 0L
}

if (!"non-V0d-cell" %in% names(count_wide)) {
  count_wide$`non-V0d-cell` <- 0L
}

Mouse_counts <- Sample_info %>%
  left_join(
    count_wide,
    by = c(
      "Slide",
      "Group",
      "Animal",
      "Sample_key"
    )
  ) %>%
  mutate(
    `V0d-cell` = replace_na(
      `V0d-cell`,
      0L
    ),

    `non-V0d-cell` = replace_na(
      `non-V0d-cell`,
      0L
    ),

    V0d = `V0d-cell`,

    nonV0d = `non-V0d-cell`,

    Total_classified =
      V0d + nonV0d,

    V0d_fraction = if_else(
      Total_classified > 0,
      V0d / Total_classified,
      NA_real_
    )
  ) %>%
  select(
    Slide,
    Group,
    Animal,
    Sample_key,
    V0d,
    nonV0d,
    Total_classified,
    V0d_fraction
  )

write_csv(
  Mouse_counts,
  file.path(
    TABLE_DIR,
    "per_mouse_counts.csv"
  )
)



# ------------------------------------------------------------
# QUANTIFICATION: FLUORESCENCE PER MOUSE
# ------------------------------------------------------------

Mouse_fluorescence <- NT %>%
  select(
    Slide,
    Group,
    Animal,
    Sample_key,
    Classification,
    all_of(
      FLUORESCENCE_COLUMNS
    )
  ) %>%
  pivot_longer(
    cols = all_of(
      FLUORESCENCE_COLUMNS
    ),
    names_to = "Marker",
    values_to = "Corrected_fluorescence"
  ) %>%
  mutate(
    Marker = sub(
      "_mean_corrected$",
      "",
      Marker
    )
  ) %>%
  group_by(
    Slide,
    Group,
    Animal,
    Sample_key,
    Classification,
    Marker
  ) %>%
  summarise(
    n_cells = sum(
      is.finite(
        Corrected_fluorescence
      )
    ),

    Median_corrected_fluorescence =
      safe_median(
        Corrected_fluorescence
      ),

    .groups = "drop"
  )

write_csv(
  Mouse_fluorescence,
  file.path(
    TABLE_DIR,
    "per_mouse_fluorescence.csv"
  )
)


# ------------------------------------------------------------
# CELL AREA PER MOUSE
# ------------------------------------------------------------

Mouse_cell_area <- NT %>%
  group_by(
    Slide,
    Group,
    Animal,
    Sample_key,
    Classification
  ) %>%
  summarise(
    n_cells = n(),

    Median_cell_area_um2 =
      median(
        Cell_area_um2,
        na.rm = TRUE
      ),

    Mean_cell_area_um2 =
      mean(
        Cell_area_um2,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

write_csv(
  Mouse_cell_area,
  file.path(
    TABLE_DIR,
    "per_mouse_cell_area.csv"
  )
)



# ------------------------------------------------------------
# STATISTICAL HELPERS
# ------------------------------------------------------------

mann_whitney_summary <- function(
  data,
  outcome,
  group_column = "Group"
) {

  keep <- data %>%
    filter(
      is.finite(
        .data[[outcome]]
      ),
      !is.na(
        .data[[group_column]]
      )
    )

  groups <- sort(
    unique(
      as.character(
        keep[[group_column]]
      )
    )
  )

  if (length(groups) != 2) {
    return(
      tibble(
        Outcome = outcome,
        Group_1 = NA_character_,
        Group_2 = NA_character_,
        n_1 = NA_integer_,
        n_2 = NA_integer_,
        Median_1 = NA_real_,
        Median_2 = NA_real_,
        W = NA_real_,
        p_value = NA_real_
      )
    )
  }

  x <- keep %>%
    filter(
      .data[[group_column]] == groups[[1]]
    ) %>%
    pull(
      all_of(outcome)
    )

  y <- keep %>%
    filter(
      .data[[group_column]] == groups[[2]]
    ) %>%
    pull(
      all_of(outcome)
    )

  if (
    length(x) == 0 ||
    length(y) == 0
  ) {
    return(
      tibble(
        Outcome = outcome,
        Group_1 = groups[[1]],
        Group_2 = groups[[2]],
        n_1 = length(x),
        n_2 = length(y),
        Median_1 = safe_median(x),
        Median_2 = safe_median(y),
        W = NA_real_,
        p_value = NA_real_
      )
    )
  }

  test <- wilcox.test(
    x,
    y,
    alternative = "two.sided",
    exact = FALSE
  )

  tibble(
    Outcome = outcome,
    Group_1 = groups[[1]],
    Group_2 = groups[[2]],
    n_1 = length(x),
    n_2 = length(y),
    Median_1 = safe_median(x),
    Median_2 = safe_median(y),
    W = unname(
      test$statistic
    ),
    p_value = test$p.value
  )
}


# ------------------------------------------------------------
# STATISTICS: COUNTS / DENSITIES
# ------------------------------------------------------------

count_outcomes <- c(
  "V0d",
  "nonV0d",
  "Total_classified",
  "V0d_fraction"
)

Count_stats <- bind_rows(
  lapply(
    count_outcomes,
    function(outcome) {
      mann_whitney_summary(
        Mouse_counts,
        outcome
      )
    }
  )
)

cat(
  "\n\n============================================================\n"
)
cat(
  "MOUSE-LEVEL COUNT / DENSITY STATISTICS\n"
)
cat(
  "Independent two-group Wilcoxon rank-sum / Mann-Whitney tests\n"
)
cat(
  "============================================================\n"
)

print(
  Count_stats,
  n = nrow(Count_stats)
)


# ------------------------------------------------------------
# STATISTICS: FLUORESCENCE
# ------------------------------------------------------------

Fluorescence_stats <- Mouse_fluorescence %>%
  group_by(
    Classification,
    Marker
  ) %>%
  group_modify(
    ~ mann_whitney_summary(
      .x,
      "Median_corrected_fluorescence"
    )
  ) %>%
  ungroup() %>%
  group_by(
    Classification
  ) %>%
  mutate(
    p_holm = p.adjust(
      p_value,
      method = "holm"
    )
  ) %>%
  ungroup()

cat(
  "\n\n============================================================\n"
)
cat(
  "MOUSE-LEVEL FLUORESCENCE STATISTICS\n"
)
cat(
  "One median fluorescence value per mouse\n"
)
cat(
  "Holm correction across the five markers within each classification\n"
)
cat(
  "============================================================\n"
)

print(
  Fluorescence_stats,
  n = nrow(Fluorescence_stats)
)


# ------------------------------------------------------------
# MOUSE-LEVEL COUNT / DENSITY FIGURES
# ------------------------------------------------------------

plot_count_endpoint <- function(
  data,
  y_column,
  y_label,
  title,
  file_name
) {

  p <- ggplot(
    data,
    aes(
      x = Group,
      y = .data[[y_column]]
    )
  ) +
    geom_boxplot(
      width = 0.45,
      outlier.shape = NA,
      fill = "white",
      linewidth = 0.7
    ) +
    geom_jitter(
      width = 0.07,
      height = 0,
      size = 2.5,
      alpha = 0.8
    ) +
    labs(
      x = NULL,
      y = y_label,
      title = title
    ) +
    theme_classic(
      base_size = 12
    )

  save_plot(
    p,
    file_name,
    width = 5,
    height = 5.5
  )
}


plot_count_endpoint(
  Mouse_counts,
  "V0d",
  "V0d cells per mouse",
  "V0d cell count",
  "V0d_raw_count"
)

plot_count_endpoint(
  Mouse_counts,
  "V0d_fraction",
  "V0d / classified cells",
  "V0d fraction",
  "V0d_fraction"
)


# ------------------------------

# ------------------------------------------------------------
# SPATIAL FIGURES — PHYSICAL COORDINATES
# ------------------------------------------------------------

for (classification in CLASSIFICATIONS) {

  spatial_data <- NT %>%
    filter(
      Classification == classification,
      is.finite(X_um),
      is.finite(Y_um)
    )

  label <- ifelse(
    classification == "V0d-cell",
    "V0d",
    "non-V0d"
  )

  p_xy <- ggplot(
    spatial_data,
    aes(
      x = X_um,
      y = Y_um
    )
  ) +
    geom_point(
      size = 0.7,
      alpha = 0.45
    ) +
    facet_wrap(
      ~ Group
    ) +
    scale_y_reverse() +
    coord_equal() +
    labs(
      x = "X position (µm)",
      y = "Y position (µm)",
      title = paste0(
        label,
        " spatial distribution"
      )
    ) +
    theme_classic(
      base_size = 12
    )

  save_plot(
    p_xy,
    paste0(
      label,
      "_physical_spatial_distribution"
    ),
    width = 8,
    height = 5
  )
}

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

cat(
  "\n\n============================================================\n"
)
cat(
  "HiPlexUp V0d analysis complete\n"
)
cat(
  "============================================================\n"
)

cat(
  "Cell rows analysed: ",
  nrow(NT),
  "\n",
  sep = ""
)

cat(
  "Unique mice/samples: ",
  n_distinct(NT$Animal),
  "\n",
  sep = ""
)

cat(
  "Groups: ",
  paste(
    sort(unique(NT$Group)),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "\nTables written to:\n  ",
  TABLE_DIR,
  "\n",
  sep = ""
)

cat(
  "Figures written to:\n  ",
  dirname(PNG_DIR),
  "\n",
  sep = ""
)

cat(
  "\nImportant: significance tests above use mouse-level summaries, ",
  "not individual cells.\n",
  sep = ""
)

cat(
  paste0(
    "\nPhysical pixel calibration supplied: ",
    PIXEL_SIZE_X_UM,
    " µm/pixel X, ",
    PIXEL_SIZE_Y_UM,
    " µm/pixel Y.\n"
  )
)

cat(
  paste0(
    "Cell coordinates are therefore available in µm and ",
    "cell areas in µm².\n"
  )
)

cat(
  paste0(
    "Exact tissue ROI masks are unavailable, so tissue-normalized ",
    "cell density (cells/mm²) is NOT calculated.\n"
  )
)