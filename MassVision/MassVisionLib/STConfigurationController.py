"""Spatial transcriptomics configuration UI controller for MassVision.

This module keeps the QTreeWidget-based advanced configuration editor,
the four promoted QLineEdit parameters, JSON load/save, and configuration
validation separate from MassVision.py.

Expected UI object names
------------------------
STconfEdit     : QTreeWidget
STpar1_val     : QLineEdit for processing["min_spots_per_gene"]
STpar2_val     : QLineEdit for processing["n_top_genes"]
STpar3_val     : QLineEdit for processing["min_genes_per_spot"]
STpar4_val     : QLineEdit for raster["target_min_dimension"]
"""

import copy
import json

import qt
import slicer

from MassVisionLib.VisiumVision import DEFAULT_PROCESSING, DEFAULT_RASTER


class STConfigurationController:
    """Manage the spatial transcriptomics configuration widgets."""

    LINE_EDIT_FIELDS = {
        "processing": {
            "min_genes_per_spot": ("STpar3_val", int),
            "min_spots_per_gene": ("STpar1_val", int),
            "n_top_genes": ("STpar2_val", int),
        },
        "raster": {
            "target_min_dimension": ("STpar4_val", int),
        },
    }

    REQUIRED_SECTIONS = ("processing", "raster")

    def __init__(self, ui):
        """
        Parameters
        ----------
        ui
            Object returned by slicer.util.childWidgetVariables(uiWidget).
        """
        self.ui = ui
        self.setupTree()

    # ------------------------------------------------------------------
    # Public API used by MassVision.py
    # ------------------------------------------------------------------

    def restoreDefaults(self, checked=False):
        """Restore and display the default configuration."""
        del checked
        self.showConfiguration(self.defaultConfiguration())

    def saveConfiguration(self, checked=False):
        """Save the complete spatial transcriptomics configuration."""
        del checked

        try:
            configuration = self.readConfiguration()
        except ValueError as error:
            slicer.util.errorDisplay(
                f"The configuration is invalid.\n\n{error}",
                windowTitle="Configuration Error",
            )
            return

        filePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save spatial transcriptomics configuration",
            "MassVision-ST-configuration.json",
            "JSON configuration (*.json)",
        )

        if not filePath:
            return

        if not filePath.lower().endswith(".json"):
            filePath += ".json"

        try:
            with open(filePath, "w", encoding="utf-8") as file:
                json.dump(
                    configuration,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )
                file.write("\n")

        except (OSError, TypeError) as error:
            slicer.util.errorDisplay(
                f"Could not save the configuration.\n\n{error}",
                windowTitle="Configuration Error",
            )

    def loadConfiguration(self, checked=False):
        """Load and display a complete JSON configuration."""
        del checked

        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Load spatial transcriptomics configuration",
            "",
            "JSON configuration (*.json);;All Files (*)",
        )

        if not filePath:
            return

        try:
            with open(filePath, "r", encoding="utf-8") as file:
                configuration = json.load(file)

            self._validateConfigurationStructure(configuration)
            self.showConfiguration(configuration)

        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            slicer.util.errorDisplay(
                f"Could not load the configuration.\n\n{error}",
                windowTitle="Configuration Error",
            )

    def getParameters(self):
        """Return complete processing and raster parameter dictionaries."""
        configuration = self.readConfiguration()

        return (
            copy.deepcopy(configuration["processing"]),
            copy.deepcopy(configuration["raster"]),
        )

    # ------------------------------------------------------------------
    # Configuration split/merge
    # ------------------------------------------------------------------

    def defaultConfiguration(self):
        """Return independent copies of the processing and raster defaults."""
        return {
            "processing": copy.deepcopy(DEFAULT_PROCESSING),
            "raster": copy.deepcopy(DEFAULT_RASTER),
        }

    def showConfiguration(self, configuration):
        """
        Display a complete configuration in the UI.

        The four promoted parameters are placed in their QLineEdit widgets
        and removed from the advanced-settings tree.
        """
        self._validateConfigurationStructure(configuration)
        self._setLineEditsFromConfiguration(configuration)

        treeConfiguration = self._removeLineEditFields(configuration)
        tree = self.ui.STconfEdit

        tree.blockSignals(True)
        try:
            tree.clear()

            for key, value in treeConfiguration.items():
                self._addConfigurationItem(tree, key, value)

            tree.expandAll()
        finally:
            tree.blockSignals(False)

    def readConfiguration(self):
        """
        Read the advanced-settings tree and merge the four QLineEdit values.
        """
        configuration = self._configurationFromTree()
        self._validateConfigurationStructure(configuration)

        return self._mergeLineEditFields(configuration)

    def _validateConfigurationStructure(self, configuration):
        """Validate the top-level configuration structure."""
        if not isinstance(configuration, dict):
            raise ValueError(
                "The top-level configuration must be a JSON object."
            )

        for section in self.REQUIRED_SECTIONS:
            if section not in configuration:
                raise ValueError(
                    f"The configuration is missing the '{section}' section."
                )

            if not isinstance(configuration[section], dict):
                raise ValueError(
                    f"The '{section}' section must contain a parameter dictionary."
                )

    def _iterLineEditFields(self):
        """Yield section, key, widget, and converter for promoted fields."""
        for section, fields in self.LINE_EDIT_FIELDS.items():
            for key, (widgetName, converter) in fields.items():
                widget = getattr(self.ui, widgetName)
                yield section, key, widget, converter

    def _setLineEditsFromConfiguration(self, configuration):
        """
        Populate promoted line edits.

        A missing promoted value in a loaded configuration falls back to
        its current default.
        """
        defaults = self.defaultConfiguration()

        for section, key, widget, _ in self._iterLineEditFields():
            value = configuration.get(section, {}).get(
                key,
                defaults[section][key],
            )
            widget.setText(str(value))

    def _removeLineEditFields(self, configuration):
        """Return a copy with promoted fields removed from the tree view."""
        treeConfiguration = copy.deepcopy(configuration)

        for section, key, _, _ in self._iterLineEditFields():
            sectionConfiguration = treeConfiguration.get(section)

            if isinstance(sectionConfiguration, dict):
                sectionConfiguration.pop(key, None)

        return treeConfiguration

    def _mergeLineEditFields(self, configuration):
        """Return a complete configuration containing the line-edit values."""
        completeConfiguration = copy.deepcopy(configuration)

        for section, key, widget, converter in self._iterLineEditFields():
            text = widget.text.strip()

            if not text:
                raise ValueError(f"'{key}' cannot be empty.")

            try:
                value = converter(text)
            except (TypeError, ValueError):
                expectedType = converter.__name__
                raise ValueError(
                    f"'{key}' must be a valid {expectedType}. "
                    f"Current value: {text!r}"
                )

            completeConfiguration.setdefault(section, {})
            completeConfiguration[section][key] = value

        return completeConfiguration

    # ------------------------------------------------------------------
    # Tree setup and dictionary-to-tree conversion
    # ------------------------------------------------------------------

    def setupTree(self):
        """Configure the editable advanced-settings QTreeWidget."""
        tree = self.ui.STconfEdit

        tree.clear()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Parameter", "Value"])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setUniformRowHeights(True)
        tree.setEditTriggers(
            qt.QAbstractItemView.DoubleClicked
            | qt.QAbstractItemView.EditKeyPressed
            | qt.QAbstractItemView.SelectedClicked
        )

        header = tree.header()
        header.setSectionResizeMode(
            0,
            qt.QHeaderView.ResizeToContents,
        )
        header.setStretchLastSection(True)

    @staticmethod
    def _humanizeParameterName(key):
        """Convert a configuration key into a readable display label."""
        return key.replace("_", " ")

    @staticmethod
    def _valueType(value):
        """Return a type identifier used when parsing an edited cell."""
        # bool must be checked before int because bool subclasses int.
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            return "list"
        if isinstance(value, tuple):
            return "tuple"
        if isinstance(value, dict):
            return "dict"

        return "json"

    @staticmethod
    def _valueToText(value):
        """Convert a Python value into editable display text."""
        if value is None:
            return "null"

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False)

        return str(value)

    def _addConfigurationItem(self, parent, key, value):
        """Recursively add one configuration entry to the tree."""
        item = qt.QTreeWidgetItem(parent)

        item.setText(0, self._humanizeParameterName(key))

        # Keep the real dictionary key even though column 0 is humanized.
        item.setData(0, qt.Qt.UserRole, key)

        valueType = self._valueType(value)
        item.setData(1, qt.Qt.UserRole, valueType)

        item.setToolTip(0, f"Configuration key: {key}")

        if isinstance(value, dict):
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)

            for childKey, childValue in value.items():
                self._addConfigurationItem(
                    item,
                    childKey,
                    childValue,
                )
            return

        item.setText(1, self._valueToText(value))
        item.setToolTip(1, f"Expected value type: {valueType}")
        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)

    # ------------------------------------------------------------------
    # Tree-to-dictionary conversion
    # ------------------------------------------------------------------

    def _parseConfigurationValue(self, text, valueType, parameterName):
        """Convert edited cell text back into a Python value."""
        text = text.strip()

        try:
            if valueType == "str":
                return text

            if valueType == "int":
                if text.lower() in ("null", "none"):
                    return None
                return int(text)

            if valueType == "float":
                if text.lower() in ("null", "none"):
                    return None
                return float(text)

            if valueType == "bool":
                normalized = text.lower()

                if normalized in ("true", "1", "yes"):
                    return True
                if normalized in ("false", "0", "no"):
                    return False
                if normalized in ("null", "none"):
                    return None

                raise ValueError("expected true or false")

            if valueType == "list":
                value = json.loads(text)

                if not isinstance(value, list):
                    raise ValueError("expected a JSON list")

                return value

            if valueType == "tuple":
                value = json.loads(text)

                if not isinstance(value, list):
                    raise ValueError("expected a JSON list")

                return tuple(value)

            if valueType == "none":
                # A nullable field may receive any valid JSON scalar/list,
                # or an unquoted string such as auto.
                if text.lower() in ("null", "none", ""):
                    return None

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text

            if valueType == "json":
                return json.loads(text)

            return text

        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid value for '{parameterName}': "
                f"{text!r}\nExpected type: {valueType}.\n{error}"
            )

    def _readConfigurationItem(self, item, path=""):
        """Recursively convert one tree item into a key-value pair."""
        key = str(item.data(0, qt.Qt.UserRole))

        parameterPath = f"{path}.{key}" if path else key
        valueType = str(item.data(1, qt.Qt.UserRole))

        if valueType == "dict":
            value = {}

            for childIndex in range(item.childCount()):
                childKey, childValue = self._readConfigurationItem(
                    item.child(childIndex),
                    parameterPath,
                )
                value[childKey] = childValue

            return key, value

        value = self._parseConfigurationValue(
            item.text(1),
            valueType,
            parameterPath,
        )

        return key, value

    def _configurationFromTree(self):
        """Return the advanced configuration currently shown in the tree."""
        configuration = {}
        root = self.ui.STconfEdit.invisibleRootItem()

        for itemIndex in range(root.childCount()):
            key, value = self._readConfigurationItem(
                root.child(itemIndex)
            )
            configuration[key] = value

        return configuration
