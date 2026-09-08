"""Lazy Zarr volume and patch iteration."""
from dataclasses import dataclass
from typing import Iterator
import numpy as np
@dataclass(slots=True)
class VolumeDataset:
    array:object
    def frame(self,t:int)->np.ndarray:return np.asarray(self.array[t])
    def patches(self,t:int,patch_size=(16,128,128),stride=(8,96,96))->Iterator[tuple[tuple[int,int,int],np.ndarray]]:
        f=self.frame(t)
        for z in range(0,max(1,f.shape[0]-patch_size[0]+1),stride[0]):
            for y in range(0,max(1,f.shape[1]-patch_size[1]+1),stride[1]):
                for x in range(0,max(1,f.shape[2]-patch_size[2]+1),stride[2]): yield (z,y,x),f[z:z+patch_size[0],y:y+patch_size[1],x:x+patch_size[2]]
def open_zarr(path:str)->VolumeDataset:
    import zarr
    return VolumeDataset(zarr.open(path,mode="r"))
