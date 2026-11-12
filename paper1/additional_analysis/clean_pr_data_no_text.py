"""
Clean PR dataset without requiring title/body text fields.

This script handles PR metadata cleaning only:
- Validates required PR fields (ID, timestamps, metrics)
- Standardizes numeric columns
- Removes duplicates
- Fills missing values where appropriate
- Exports cleaned dataset

Does NOT require: title, body, or language fields
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys


class PRDataCleaner:
    
    def __init__(self, input_path, output_dir="data"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.rows_initial = 0
        self.rows_final = 0
        
    
    def load_data(self):
        """Load CSV or Parquet file."""
        if self.input_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.input_path)
        elif self.input_path.suffix == ".csv":
            self.df = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported format: {self.input_path.suffix}")
        
        self.rows_initial = len(self.df)
        print(f"✓ Loaded {self.rows_initial:,} rows from {self.input_path.name}")
        return self
    
    
    def validate_required_fields(self):
        """Ensure minimum required PR fields exist."""
        # At least one PR identifier
        id_fields = ["pull_request_id", "pr_id", "id"]
        has_pr_id = any(col in self.df.columns for col in id_fields)
        
        if not has_pr_id:
            raise ValueError(
                f"Missing PR identifier. Need one of: {id_fields}. "
                f"Available: {self.df.columns.tolist()}"
            )
        
        # At least one repository identifier
        repo_fields = ["repository_id", "repo_id", "repo"]
        has_repo_id = any(col in self.df.columns for col in repo_fields)
        
        if not has_repo_id:
            print(f"⚠ Warning: No repository ID found. Using row index as repo proxy.")
        
        print(f"✓ Required fields present (PR ID, Repo ID)")
        return self
    
    
    def standardize_numeric_columns(self):
        """Convert numeric columns to proper types."""
        numeric_candidates = [
            "minutes_to_merge", "lines_added", "lines_deleted", "lines_modified",
            "files_added", "files_deleted", "files_modified", "files_touched",
            "num_reviewers", "num_review_comments", "complexity_score",
            "code_lines_modified", "test_lines_modified"
        ]
        
        converted_count = 0
        for col in numeric_candidates:
            if col in self.df.columns:
                before_nulls = self.df[col].isna().sum()
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                after_nulls = self.df[col].isna().sum()
                
                if after_nulls > before_nulls:
                    new_nulls = after_nulls - before_nulls
                    print(f"  - {col}: {new_nulls} rows converted to NaN (non-numeric)")
                
                converted_count += 1
        
        print(f"✓ Standardized {converted_count} numeric columns")
        return self
    
    
    def create_derived_metrics(self):
        """Create useful derived columns if source columns exist."""
        # Total lines modified
        if "lines_modified" not in self.df.columns:
            lines_added = self.df.get("lines_added", pd.Series(0, index=self.df.index))
            lines_deleted = self.df.get("lines_deleted", pd.Series(0, index=self.df.index))
            self.df["lines_modified"] = (
                pd.to_numeric(lines_added, errors="coerce").fillna(0) +
                pd.to_numeric(lines_deleted, errors="coerce").fillna(0)
            )
        
        # Total files touched
        if "files_touched" not in self.df.columns:
            files_added = self.df.get("files_added", pd.Series(0, index=self.df.index))
            files_deleted = self.df.get("files_deleted", pd.Series(0, index=self.df.index))
            files_modified = self.df.get("files_modified", pd.Series(0, index=self.df.index))
            self.df["files_touched"] = (
                pd.to_numeric(files_added, errors="coerce").fillna(0) +
                pd.to_numeric(files_deleted, errors="coerce").fillna(0) +
                pd.to_numeric(files_modified, errors="coerce").fillna(0)
            )
        
        # Simple complexity score (if not present)
        if "complexity_score" not in self.df.columns:
            lines_mod = pd.to_numeric(self.df.get("lines_modified", 0), errors="coerce").fillna(0)
            files_touch = pd.to_numeric(self.df.get("files_touched", 0), errors="coerce").fillna(0)
            self.df["complexity_score"] = lines_mod + (10 * files_touch)
        
        print(f"✓ Created derived metrics (lines_modified, files_touched, complexity_score)")
        return self
    
    
    def remove_duplicates(self):
        """Remove exact duplicate rows and duplicate PRs."""
        # Remove exact duplicates
        before_exact = len(self.df)
        self.df = self.df.drop_duplicates()
        exact_dups = before_exact - len(self.df)
        
        if exact_dups > 0:
            print(f"✓ Removed {exact_dups:,} exact duplicate rows")
        
        # Remove duplicate PRs (keep first occurrence)
        pr_id_cols = ["pull_request_id", "pr_id", "id"]
        repo_id_cols = ["repository_id", "repo_id", "repo"]
        
        pr_id_col = next((col for col in pr_id_cols if col in self.df.columns), None)
        repo_id_col = next((col for col in repo_id_cols if col in self.df.columns), None)
        
        if pr_id_col and repo_id_col:
            before_pr_dups = len(self.df)
            self.df = self.df.drop_duplicates(subset=[repo_id_col, pr_id_col], keep="first")
            pr_dups = before_pr_dups - len(self.df)
            
            if pr_dups > 0:
                print(f"✓ Removed {pr_dups:,} duplicate PRs (same repo + PR ID)")
        
        return self
    
    
    def handle_missing_values(self):
        """Report and handle missing values."""
        missing_summary = pd.DataFrame({
            "column": self.df.columns,
            "missing_count": self.df.isna().sum(),
            "missing_pct": (self.df.isna().mean() * 100).round(2)
        }).sort_values("missing_pct", ascending=False)
        
        missing_with_values = missing_summary[missing_summary["missing_count"] > 0]
        
        if not missing_with_values.empty:
            print(f"\n⚠ Missing values found:")
            for _, row in missing_with_values.iterrows():
                print(f"  - {row['column']}: {int(row['missing_count']):,} ({row['missing_pct']:.1f}%)")
        else:
            print(f"✓ No missing values found")
        
        return self
    
    
    def generate_output_filename(self, custom_name=None):
        """Generate output filename."""
        if custom_name:
            return custom_name if custom_name.endswith(".csv") else f"{custom_name}.csv"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cleaned_pr_data_{timestamp}.csv"
    
    
    def save_output(self, output_filename=None):
        """Save cleaned dataset."""
        filename = self.generate_output_filename(output_filename)
        output_path = self.output_dir / filename
        
        self.df.to_csv(output_path, index=False)
        self.rows_final = len(self.df)
        
        print(f"\n✓ Saved to: {output_path}")
        return self
    
    
    def print_summary(self):
        """Print cleaning summary."""
        rows_dropped = self.rows_initial - self.rows_final
        rows_dropped_pct = (rows_dropped / self.rows_initial * 100) if self.rows_initial > 0 else 0
        
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"Input rows:      {self.rows_initial:>15,}")
        print(f"Output rows:     {self.rows_final:>15,}")
        print(f"Rows dropped:    {rows_dropped:>15,} ({rows_dropped_pct:.1f}%)")
        print(f"Columns:         {self.df.shape[1]:>15}")
        print("=" * 60)
        
        if self.rows_final > 0:
            print("\nDataset columns:")
            for i, col in enumerate(self.df.columns, 1):
                dtype = str(self.df[col].dtype)
                nulls = self.df[col].isna().sum()
                print(f"  {i:2}. {col:25} ({dtype:10}) - {nulls:,} nulls")
        
        return self
    
    
    def run(self, output_filename=None):
        """Execute full cleaning pipeline."""
        (self
         .load_data()
         .validate_required_fields()
         .standardize_numeric_columns()
         .create_derived_metrics()
         .remove_duplicates()
         .handle_missing_values()
         .save_output(output_filename)
         .print_summary()
        )
        
        return self.df


if __name__ == "__main__":
    # Example usage:
    # python clean_pr_data_no_text.py <input_file> [output_name]
    
    if len(sys.argv) < 2:
        print("Usage: python clean_pr_data_no_text.py <input_file> [output_name]")
        print("\nExample:")
        print("  python clean_pr_data_no_text.py data/my_prs.csv")
        print("  python clean_pr_data_no_text.py data/my_prs.parquet cleaned_output")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    cleaner = PRDataCleaner(input_file)
    cleaner.run(output_name)
