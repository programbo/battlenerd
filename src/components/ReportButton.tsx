import React, { useState, useEffect } from "react";
import { DataRequestModal } from "./DataRequestModal";
import { ValidationResult } from "../types/validation";

interface ReportButtonProps {
  templateName: string;
  templateNames: string[]; // Array of related templates to validate
  context: Record<string, string>;
  onGenerate: () => void;
}

export const ReportButton: React.FC<ReportButtonProps> = ({
  templateName,
  templateNames,
  context,
  onGenerate,
}) => {
  const [validations, setValidations] = useState<
    Record<string, ValidationResult>
  >({});
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const validateTemplates = async () => {
    setLoading(true);
    try {
      const response = await fetch("/validate-templates", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          template_names: templateNames,
          available_context: context,
        }),
      });
      const data = await response.json();
      setValidations(data.templates);
    } catch (error) {
      console.error("Validation failed:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    validateTemplates();
  }, [templateNames.join(","), context]);

  const getCurrentValidation = () => validations[templateName];

  const getButtonStyle = () => {
    const validation = getCurrentValidation();
    if (!validation) return "btn-secondary";
    if (!validation.can_generate) return "btn-danger";
    if (validation.completion_percentage < 80) return "btn-warning";
    return "btn-success";
  };

  const getButtonText = () => {
    const validation = getCurrentValidation();
    if (loading) return "Checking...";
    if (!validation) return "Generate Report";
    return `Generate Report (${Math.round(
      validation.completion_percentage
    )}% Complete)`;
  };

  const handleClick = () => {
    const validation = getCurrentValidation();
    if (validation?.missing_critical.length) {
      setShowModal(true);
    } else {
      onGenerate();
    }
  };

  return (
    <div>
      <button
        className={`btn ${getButtonStyle()}`}
        onClick={handleClick}
        disabled={loading || !getCurrentValidation()?.can_generate}
      >
        {getButtonText()}
      </button>
      {getCurrentValidation()?.warning_message && (
        <div className="text-warning mt-2">
          {getCurrentValidation()?.warning_message}
        </div>
      )}
      {showModal && (
        <DataRequestModal
          requests={getCurrentValidation()?.data_requests || []}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
};
