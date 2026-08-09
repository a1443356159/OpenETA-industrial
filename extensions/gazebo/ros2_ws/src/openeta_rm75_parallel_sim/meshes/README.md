# Mesh resources

The RM75 mesh files have a single source of truth in
`extensions/gazebo/assets/rm75_6fb_v_vendor/meshes`. The shared
`openeta_rm75_v_description` package copies that embedded closure into its
installed package share directory. This
directory is intentionally kept free of a second mesh copy so relocated builds
cannot silently diverge from the manifest-verified vendor assets.
