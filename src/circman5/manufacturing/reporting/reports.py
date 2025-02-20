# src/circman5/manufacturing/reporting/reports.py

import pandas as pd
from typing import Dict, Optional, Union
from pathlib import Path
from ...utils.results_manager import results_manager
from ...utils.logging_config import setup_logger
from ...utils.errors import ProcessError


class ReportGenerator:
    def __init__(self):
        self.logger = setup_logger("manufacturing_reports")
        self.reports_dir = results_manager.get_path("reports")

    def generate_comprehensive_report(
        self, metrics: Dict, output_dir: Optional[Path] = None
    ) -> None:
        """Generate a comprehensive analysis report including all metrics."""
        try:
            # If output_dir is a file path, use it directly
            if output_dir and output_dir.suffix == ".xlsx":
                output_path = output_dir
            else:
                # If it's a directory or None, append filename
                base_dir = output_dir if output_dir else self.reports_dir
                output_path = base_dir / "comprehensive_analysis.xlsx"

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(str(output_path)) as writer:
                for metric_type, data in metrics.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)

            self.logger.info(f"Comprehensive report generated at: {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise ProcessError(f"Report generation failed: {str(e)}")

    def export_analysis_report(
        self, report_data: Dict, output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Generate and export comprehensive analysis report."""
        try:
            if output_path is None:
                output_path = self.reports_dir / "analysis_report.xlsx"

            with pd.ExcelWriter(str(output_path)) as writer:
                has_data = False
                for metric_type, data in report_data.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)
                        has_data = True

                if not has_data:
                    pd.DataFrame(["No data available"]).to_excel(
                        writer, sheet_name="Empty_Report"
                    )

            self.logger.info(f"Analysis report exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {str(e)}")
            raise ProcessError(f"Report export failed: {str(e)}")

    def generate_lca_report(
        self, impact_data: Dict, batch_id: Optional[str] = None
    ) -> None:
        """Generate comprehensive LCA report."""
        try:
            output_path = (
                self.reports_dir
                / f"lca_report{'_' + batch_id if batch_id else ''}.xlsx"
            )

            with pd.ExcelWriter(str(output_path)) as writer:
                for category, data in impact_data.items():
                    pd.DataFrame([data]).to_excel(writer, sheet_name=category)

            self.logger.info(f"LCA report generated successfully at {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating LCA report: {str(e)}")
            raise ProcessError(f"LCA report generation failed: {str(e)}")

    def save_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Save performance metrics."""
        try:
            output_path = self.reports_dir / "performance_metrics.xlsx"
            pd.DataFrame([metrics]).to_excel(str(output_path))
            self.logger.info(f"Performance metrics saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
            raise ProcessError(f"Performance metrics save failed: {str(e)}")

    def generate_performance_report(
        self, metrics: Dict[str, float], save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Generate visual performance report."""
        try:
            if save_path is None:
                save_path = self.reports_dir / "performance_report.xlsx"

            excel_data = {
                "Overall Metrics": pd.DataFrame([metrics]),
                "Detailed Analysis": pd.DataFrame(
                    [
                        {
                            "Manufacturing Efficiency": metrics.get("efficiency", 0),
                            "Quality Score": metrics.get("quality_score", 0),
                            "Resource Efficiency": metrics.get(
                                "resource_efficiency", 0
                            ),
                            "Energy Efficiency": metrics.get("energy_efficiency", 0),
                        }
                    ]
                ),
            }

            with pd.ExcelWriter(str(save_path)) as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name)

            self.logger.info(f"Performance report generated at: {save_path}")

        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            raise ProcessError(f"Performance report generation failed: {str(e)}")
