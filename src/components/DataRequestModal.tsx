interface DataRequestModalProps {
  requests: DataRequest[];
  onClose: () => void;
}

const DataRequestModal: React.FC<DataRequestModalProps> = ({
  requests,
  onClose,
}) => {
  return (
    <div className="modal">
      <div className="modal-content">
        <h3>Additional Information Needed</h3>
        <p>
          Please provide the following information to generate a complete
          report:
        </p>

        {requests
          .sort((a, b) => a.priority - b.priority)
          .map((request) => (
            <div key={request.field} className="data-request">
              <h4>
                Priority {request.priority}: {request.field}
              </h4>
              <p>{request.justification}</p>
              {request.examples && (
                <div className="examples">
                  <small>Examples:</small>
                  <ul>
                    {request.examples.map((example, i) => (
                      <li key={i}>{example}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  );
};
