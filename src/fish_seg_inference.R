library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
plotColor <- function(imgList, scale=TRUE, vectors=NULL, points=NULL, paths=NULL) {

  if (class(imgList) == "antsImage") {
    if ( imgList@components > 1 ) imgList = splitChannels( imgList )
    else if ( imgList@components == 1 ) imgList = list(imgList, imgList, imgList)
  }

  direction = antsGetDirection( imgList[[1]] )

  # max in all images
  maxi = 1.0
  if ( scale )
    {
    maxi = max( unlist( lapply( imgList, function(x) { max(x) } ) ) )
    }

  rgbList = lapply( imgList, function(x) { apply(t(as.matrix(x)),2,rev) / maxi })
  rgbList = lapply( imgList, function(x) { t(as.matrix(x)) / maxi })

  col <- rgb(rgbList[[1]], rgbList[[2]], rgbList[[3]])

  d = dim(rgbList[[1]])

  x = rep(1:d[2],each=d[1])
  y = rep(1:d[1], d[2])
  pts = antsTransformIndexToPhysicalPoint( imgList[[1]], cbind(x,y) )

  dat = data.frame(x=pts[,1], y=pts[,2], col=col)
  x1 = min(pts[,1])
  x2 = max(pts[,1])
  y1 = min(pts[,2])
  y2 = max(pts[,2])

  g = ggplot(dat) + geom_raster(aes(x=x, y=y, fill=col), hjust=0, vjust=0, alpha=1) + theme(legend.position="none", aspect.ratio=1,text=element_blank(),axis.ticks=element_blank(), panel.grid=element_blank() ) + scale_fill_manual(values=as.character(levels(factor(col))) )

  g = g + coord_cartesian( xlim=c(x1,x2), ylim=c(y1,y2) )
  if ( direction[1,1] > 0 ) {
    g = g + scale_x_continuous( lim=c(x1,x2) )
    }
  else {
    g = g + scale_x_reverse( lim=c(x2,x1) )
   }
  if ( direction[2,2] > 0 ) {
    g = g + scale_y_continuous( lim=c(y1,y2) )
    }
  else {
    g = g + scale_y_reverse( lim=c(y2,y1) )
   }

  if ( !is.null(points) ) {
    pdat = data.frame( x=points[,1], y=points[,2], id=factor(1:dim(points)[1]) )
    g = g + geom_point( data=pdat, aes(x=x, y=y, colour=id ))
  }

  if ( !is.null(paths) ) {
    g = g + geom_path(data=paths, aes(x=x,y=y,group=id,colour=id))
    }

  if ( !is.null(vectors) ) {
    xvec = as.vector( t(as.matrix(vectors[[1]])) )
    yvec = as.vector( -t(as.matrix(vectors[[2]])) )
    vpts = antsTransformIndexToPhysicalPoint( imgList[[1]], cbind(x+0.5,y+0.5) )

    mag = sqrt(xvec*xvec + yvec*yvec)
    elim = which(mag < 0.01)
    if (length(elim) > 0 ) {
      xvec = xvec[-elim]
      yvec = yvec[-elim]
      vpts = vpts[-elim,]
      }
    vdat = data.frame(x=vpts[,1]-xvec, y=vpts[,2]-yvec, xend=vpts[,1]+xvec, yend=vpts[,2]+yvec)
    g = g + geom_segment(data=vdat, aes(x=x,y=y,xend=xend,yend=yend), colour="red", alpha=0.5)
  }

  suppressWarnings(print(g))
}

# choose the images you will segment
imageFNS = Sys.glob( "images/*jpg" )
imageFNS = Sys.glob( "unseen_test/*jpg" )
labelIDs = read.csv( "src/label_identities.csv")
nChannels = 3
remapSegmentation <- function( x ) {
  # simplify segmentations into common labels that can reliably be segmented
  newseg = x * 0
  for ( k in 1:nrow( labelIDs ) ) {
    y = labelIDs$LabelNumber[k]
    z = labelIDs$JoinWith[k]
    newseg = newseg + thresholdImage( x, y, y ) * z
  }
  newseg
}

read.fcsv<-function( x, toLPS = TRUE, skip=3 ) {
  df = read.table( x, skip=skip, sep=',' )[,c(1:3,12)]
  colnames( df ) = c("id","x","y",'Label')
  return( df )
  }

polarX <- function(X) {
        x_svd <- svd(X)
        P <- x_svd$u %*% diag(x_svd$d) %*% t(x_svd$u)
        Z <- x_svd$u %*% t(x_svd$v)
        if (det(Z) < 0)
            Z = Z * (-1)
        return(list(P = P, Z = Z, Xtilde = P %*% Z))
    }

randAff <- function( loctx,  txtype = "Rigid", sdAffine,
      idparams, fixParams, seed ) {
      set.seed( seed )
      idim = 2
      noisemat = stats::rnorm(length(idparams), mean = 0, sd = sdAffine)
      if (txtype == "Translation")
        noisemat[1:(length(idparams) - idim )] = 0
      idparams = idparams + noisemat
      idmat = matrix(idparams[1:(length(idparams) - idim )],
                  ncol = idim )
      idmat = polarX(idmat)
              if (txtype == "Rigid")
                  idmat = idmat$Z
              if (txtype == "Affine")
                  idmat = idmat$Xtilde
              if (txtype == "ScaleShear")
                  idmat = idmat$P
              idparams[1:(length(idparams) - idim )] = as.numeric(idmat)
      setAntsrTransformParameters(loctx, idparams)
      setAntsrTransformFixedParameters( loctx, fixParams )
      return(loctx)
      }
randomRotateImage <- function( image, sdAff=0.1, seed = 0 ) {
  fixedParams = getCenterOfMass( image * 0 + 1 )
  loctx <- createAntsrTransform(precision = "float",
    type = "AffineTransform", dimension = image@dimension  )
  setAntsrTransformFixedParameters(loctx, fixedParams)
  idparams = getAntsrTransformParameters( loctx )
  setAntsrTransformParameters( loctx, idparams )
  setAntsrTransformFixedParameters(loctx, fixedParams)
  loctx = randAff( loctx, sdAffine=sdAff, txtype = 'Affine',
    idparams = idparams, fixParams = fixedParams, seed = seed )
  imageR = applyAntsrTransformToImage( loctx, image, image,
      interpolation = "nearestNeighbor" )
  return( imageR )
}


generateData <- function( imgIn,
  batch_size = 16, mySdAff=0.15,
  subSampling = 8, doReflect=FALSE ) {
  reflectMat = '/tmp/reflect.mat'
  imgSub = resampleImage( imgIn, dim( imgIn )/subSampling, useVoxels=T)
#  lummsk = getMask( mylum, mean( mylum ) * 0.5, max( mylum ) * 0.95 ) %>% morphology("dilate",10)
  # thresholdImage( mylum, quantile(mylum,0.1), quantile(mylum,0.99) )
  temp = splitChannels( imgSub )
  for ( k in 1:nChannels ) {
#    temp[[k]] = n3BiasFieldCorrection( temp[[k]], 4  )
    temp[[k]] = iMath(temp[[k]], "PadImage", 0 ) %>% ANTsRNet::padImageByFactor( 8 )
    if ( k == 1 ) myreflection = reflectionMatrix( temp[[k]], 0, reflectMat )
    if ( doReflect ) temp[[k]] = antsApplyTransforms( temp[[k]], temp[[k]], reflectMat )
    }
  imgSub = mergeChannels( temp )
  mylum = toLuminance(  imgSub )
  X = array( dim = c( batch_size, dim( imgSub  ), nChannels ) )
  for ( k in 1:batch_size ) {
    seeder = Sys.time()
    splitter = splitChannels( imgSub )
    for ( j in 1:nChannels ) {
      rr = randomRotateImage( splitter[[j]], sdAff=mySdAff, seed = seeder ) %>%
        iMath( "Normalize" ) # %>% iMath( "TruncateIntensity", 1e-9, 0.9 )
      if ( imgSub@dimension == 2 ) X[k,,,j] = as.array( rr  )
      if ( imgSub@dimension == 3 ) X[k,,,,j] = as.array( rr  )
    }
  }
  return( list( X, mylum ) )
}

toLuminance <- function( x ) {
  temp = splitChannels( x )
  w = c(0.299,0.587,0.114)
  temp =  temp[[1]]*w[1]+temp[[2]]*w[2]+temp[[3]]*w[3]
  msk = getMask( temp, 5, 253 )
  temp2 = -( temp * msk )
  iMath( temp2, "Normalize" ) * 255 * msk
  }


######## here, we find the unique labels #########
ulabs = sort( unique( c( 0,  labelIDs$JoinWith ) ) )
if ( ! exists( "ulabs" ) ) {
  fns=Sys.glob("labelmaps/*")
  ulabs=sort(unique(antsImageRead( fns[1] )))
  for ( k in 1:length(fns) ) ulabs=intersect( ulabs, unique(antsImageRead( fns[1] )) )
}


unetfn = "models/fish_seg.h5"
csvfn = gsub(  "h5", "csv" , unetfn )

if ( file.exists( unetfn ) ) {
  out <- tryCatch(
    {
    unet = load_model_hdf5( unetfn, compile=FALSE  )
    },
    error=function(cond) {
        return(NA)
    },
    warning=function(cond) {
        return(NULL)
    },
    finally={
      unet = load_model_hdf5( unetfn, compile=FALSE  )
    } )
} else message("cannot find model file")
set.seed( Sys.time() )
nums = 4
nums = sample( 1:length( imageFNS ) )
nums = 1:length( imageFNS )
# layout( matrix( 1:10, nrow = 2, byrow=F ) )
for ( kk in  nums ) {
  img = antsImageRead( imageFNS[kk] )
  limg = toLuminance(img)
  limg[ limg > 250 ] = 0
  mySubSam = ( dim( img ) / 400 )
  # choice of this parameter can have a strong effect on outcome
  # below, we use a guess at an automated scaling approach
  mySubSamR = round( mySubSam^2 )
  mySubSam = min(mySubSamR)
  gg = generateData( img, subSampling = mySubSam, batch_size = 1,
    mySdAff = 0.0, doReflect=FALSE  )
  refimg = gg[[2]]
  pp = predict( unet, tf$cast( gg[[1]], "float32" ) )
  dd = decodeUnet( pp, refimg )
  for ( jj in 1:length(dd) ) dd[[1]][[jj]] = resampleImageToTarget( dd[[1]][[jj]], limg )
  segvec = imageListToMatrix( dd[[1]], limg * 0 + 1 )
  segvec = apply( segvec, FUN=which.max, MARGIN=2 )
  usegimg = makeImage( limg * 0 + 1, segvec )
  if ( !exists( "outdir" ) ) {
    outdir = './segmentation_results/'
    dir.create( outdir, showWarnings = F, recursive = T )
  }
  outname = tools::file_path_sans_ext( basename( imageFNS[kk] ) )
  ofn = paste0( outdir, outname, "_segmentation.nii.gz" )
  antsImageWrite( usegimg, ofn )
  ofn = paste0( outdir, outname, "_luminance.nii.gz" )
  antsImageWrite( limg, ofn )
#  plot( limg  )
#  plot( limg, usegimg, window.overlay=c(2,max(usegimg)), alpha=0.5 )
#  for ( jj in 1:9) plot(refimg, dd[[1]][[jj]] )
  }
