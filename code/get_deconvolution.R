library(spacexr)
library(anndata)
library(stringr)
library(Matrix)
library(biomaRt)
library(reticulate)
library(optparse)
parser <- OptionParser()
parser <- add_option(parser, c("-p", "--project_path"), type="character", default="./",
                    help="Main directory path [default %default]", metavar="character")
parser <- add_option(parser, c("-o", "--output_path"), type="character", default="AURORA_interim",
                    help="Output directory [default %default]", metavar="character")
parser <- add_option(parser, c("-m", "--metadata_path"), type="character", default="Sample_metadata.csv",
                    help="Sample metadata CSV file name [default %default]", metavar="character")
parser <- add_option(parser, c("-d", "--processed_data_path"), type="character", default="processed_data",
                    help="Processed data directory path [default %default]", metavar="character")
parser <- add_option(parser, c("-r", "--reference_file"), type="character", default="RCTD_reference.rds",
                    help="RCTD reference RDS file name [default %default]", metavar="character")
parser <- add_option(parser, c("--python_path"), type="character", default=NULL,
                    help="Path to the python executable [default %default]", metavar="character")
parser <- add_option(parser, c("-n", "--num_cores"), type="integer", default=10,
                    help="Number of cores to use [default %default]", metavar="integer")
opt <- parse_args(parser)

if (!is.null(opt$python_path)){
    use_python(opt$python_path)
}
plot_weights <- function(cell_type_names, puck, resultsdir, weights) {
  plots <- vector(mode = "list", length = length(cell_type_names))
  for (i in 1:length(cell_type_names)) {
    cell_type = cell_type_names[i]
    my_cond = weights[,cell_type] > spacexr:::UMI_cutoff(puck@nUMI) # pmax(0.25, 0.7 - puck@nUMI / 1000)
    plot_var <- weights[,cell_type]; names(plot_var) = rownames(weights)
    if(sum(my_cond) > 0)
      plots[[i]] <- plot_puck_wrapper(puck, plot_var, NULL, minUMI = 100,maxUMI = 200000,min_val = 0, max_val = 1, title = cell_type, my_cond = my_cond)
  }
  pdf(file.path(resultsdir))
  invisible(lapply(plots, print))
  dev.off()
}

main_dir = opt$project_path
st_folder = paste0(main_dir,'/',opt$processed_data_path,"/st/")
meta = read.csv(paste0(main_dir,"/",opt$metadata_path), header = T, row.names = 1)
all_samples = rownames(meta)

## make new directory if not exist
if (!dir.exists(paste0(main_dir,'/',opt$output_path,"/deconvolution/"))){
    dir.create(paste0(main_dir,'/',opt$output_path,"/deconvolution/"), showWarnings = FALSE)
}

## Build reference
## Kim et al. in https://www.weizmann.ac.il/sites/3CA/Lung
## https://www.sciencedirect.com/science/article/pii/S1535610821001173
if (!dir.exists(paste0(main_dir,'/',opt$output_path, "/deconvolution/RCTD_ref.rds"))){
    sc_ref <- readRDS(paste0(main_dir,"/",opt$reference_file))
    sc_ref <- sc_ref[,sc_ref$cell_type != ""]
    cell_types_major <- factor(sc_ref$cell_type)
    # cell_types_minor <- factor(sc_ref$celltype_minor)
    rownames(sc_ref@assays$RNA@layers$counts) <- rownames(sc_ref)
    colnames(sc_ref@assays$RNA@layers$counts) <- colnames(sc_ref)
    sc_ref <- sc_ref@assays$RNA@layers$counts
    write.table(cbind(levels(cell_types_major)), paste0(main_dir,'/',opt$output_path,"/deconvolution/cell_types.csv"), sep = ",", quote=F, row.names = F, col.names = F)

    reference_major <- Reference(sc_ref, cell_types_major, min_UMI=10)
    saveRDS(reference_major, paste0(main_dir,'/',opt$output_path,"/deconvolution/RCTD_ref.rds"))
}else{
    reference_major <- readRDS(paste0(main_dir,'/',opt$output_path, "/deconvolution/RCTD_ref.rds"))
}


if (!file.exists(paste0(main_dir,"/gene_symbols.csv"))){
    ensembl = useMart("ensembl", dataset = "hsapiens_gene_ensembl") # For human (hsapiens)
    gene_symbols = getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
               mart = ensembl)
    write.csv(gene_symbols,paste0(main_dir,"/gene_symbols.csv"),quote=F)
}else{
    gene_symbols = read.csv(paste0(main_dir,"/gene_symbols.csv"), header = T)
}

for (sample in all_samples){
    sample_name <- sample

    if (file.exists(paste0(main_dir,'/',opt$output_path,"/deconvolution/",sample_name,"_prop.csv"))){
        print(paste0("Deconvolution for ", sample_name, " already done. Skipping..."))
        next
    }

    ad <- read_h5ad(paste0(st_folder, sample, ".h5ad"))

    # covert ensembl gene names to gene symbols if needed
    if (str_starts(ad$var_names[1],"ENSG")){
        expression_data <- data.frame(t(ad$X))
        expression_data$ENSG_ID <- rownames(expression_data)
        expression_data <- merge(expression_data, gene_symbols, by.x = "ENSG_ID", by.y = "ensembl_gene_id", all.x = TRUE)
        expression_data <- expression_data[!is.na(expression_data$hgnc_symbol),]
        expression_data_summed <- aggregate(expression_data[, 2:(ncol(expression_data)-1)], by = list(expression_data$hgnc_symbol), FUN = sum)
        colnames(expression_data_summed)[1] <- "hgnc_symbol"
        rownames(expression_data_summed) <- expression_data_summed$hgnc_symbol
        expression_data_summed <- expression_data_summed[,-1]
        expression_data_summed <- expression_data_summed[rownames(expression_data_summed) != "",]
        exp_use <- as.matrix(expression_data_summed)
        colnames(exp_use) <- str_remove(colnames(exp_use), "X")
        rm(expression_data, expression_data_summed)
    } else{
        exp_use = t(ad$X)
    }

    ad_loc <- ad$obsm$spatial
    ad_loc <- as.data.frame(ad_loc)
    rownames(ad_loc) <- ad$obs_names
    RCTD_obj <- SpatialRNA(ad_loc, exp_use)
    barcodes <- colnames(RCTD_obj@counts) # pixels to be used (a list of barcode names). 
    
    #deconvolute major cell types
    myRCTD <- create.RCTD(RCTD_obj, reference_major, max_cores = opt$num_cores)
    myRCTD <- run.RCTD(myRCTD, doublet_mode = 'full')
    results <- myRCTD@results
    # normalize the cell type proportions to sum to 1.
    norm_weights = normalize_weights(results$weights) 
    rownames(norm_weights) <- colnames(myRCTD@spatialRNA@counts)
    cell_type_names <- myRCTD@cell_type_info$info[[2]]
    spatialRNA <- myRCTD@spatialRNA
    norm_weights = as.matrix(norm_weights)
    write.csv(norm_weights,paste0(main_dir,'/',opt$output_path,"/deconvolution/",sample_name,"_prop.csv"),quote=F)
    plot_weights(cell_type_names, spatialRNA, paste0(main_dir,'/',opt$output_path,"/deconvolution/",sample_name,"_prop.pdf"), norm_weights)
}

