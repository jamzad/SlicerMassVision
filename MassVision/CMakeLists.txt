#-----------------------------------------------------------------------------
set(MODULE_NAME MassVision)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__
  ${MODULE_NAME}Lib/Logic
  ${MODULE_NAME}Lib/Utils
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/alignment.png
  Resources/Icons/dataset.png
  Resources/Icons/file.png
  Resources/Icons/home.png
  Resources/Icons/inference.png
  Resources/Icons/marker.png
  Resources/Icons/preprocess.png
  Resources/Icons/report.png
  Resources/Icons/roi.png
  Resources/Icons/stat.png
  Resources/Icons/train.png
  Resources/Icons/UI_nameM.png
  Resources/Icons/visualization.png
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

endif()
